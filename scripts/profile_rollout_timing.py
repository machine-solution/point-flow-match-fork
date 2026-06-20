#!/usr/bin/env python3
"""Short rollout timing diagnostic for FM baseline vs MeanFlow multistep vs Shortcut."""
from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from omegaconf import OmegaConf, open_dict

_REPO = Path(__file__).resolve().parents[1]
_dp = _REPO.parent / "diffusion_policy"
if _dp.is_dir() and str(_dp) not in sys.path:
    sys.path.insert(0, str(_dp))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from pfp import REPO_DIRS, set_seeds  # noqa: E402
from pfp.envs.rlbench_runner import PROFILE_CSV_COLUMNS, RLBenchRunner  # noqa: E402
from pfp.policy.base_policy import BasePolicy  # noqa: E402


@dataclass
class MethodSpec:
    name: str
    ckpt_name: str
    ckpt_episode: str = "latest"
    meanflow_multistep: bool = False


METHODS: dict[str, MethodSpec] = {
    "baseline": MethodSpec("baseline", "1779122560-baseline-many-ckpts", "ep1500"),
    "meanflow_multistep": MethodSpec(
        "meanflow_multistep", "meanflow_open_fridge_1365", "latest", meanflow_multistep=True
    ),
    "shortcut": MethodSpec("shortcut", "shortcut_open_fridge_1385", "latest"),
}


def _configure_runtime_env() -> None:
    # Must match bash/run_validate_accuracy.sh: PyRep + CoppeliaSim 4.10 often heap-corrupts.
    cs41 = REPO_DIRS.ROOT / "CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
    if cs41.is_dir():
        os.environ["COPPELIASIM_ROOT"] = str(cs41)
        os.environ["LD_LIBRARY_PATH"] = f"{cs41}:{os.environ.get('LD_LIBRARY_PATH', '')}"
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(cs41)
    dp = (REPO_DIRS.ROOT.parent / "diffusion_policy").resolve()
    if dp.is_dir():
        os.environ["PYTHONPATH"] = f"{dp}:{REPO_DIRS.ROOT}:{os.environ.get('PYTHONPATH', '')}"
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")


def _load_policy(spec: MethodSpec, *, k: int) -> BasePolicy:
    ckpt_path = REPO_DIRS.CKPT / spec.ckpt_name
    train_cfg = OmegaConf.load(ckpt_path / "config.yaml")
    policy_class = hydra.utils.get_class(train_cfg.model._target_)
    load_kwargs = {
        "ckpt_name": spec.ckpt_name,
        "ckpt_episode": spec.ckpt_episode,
        "num_k_infer": int(k),
        "flow_schedule": getattr(train_cfg.model, "flow_schedule", None),
        "exp_scale": getattr(train_cfg.model, "exp_scale", None),
        "subs_factor": 1,
        "phase_conditioning": getattr(train_cfg, "phase_conditioning", None),
        "phase_prediction": getattr(train_cfg, "phase_prediction", None),
        "phase_rollout": getattr(train_cfg, "phase_rollout", None),
    }
    sig = inspect.signature(policy_class.load_from_checkpoint)
    if spec.meanflow_multistep and hasattr(policy_class, "set_meanflow_multistep_infer"):
        # Avoid one-step warning during load; enable multistep before num_k_infer is applied.
        load_kwargs["num_k_infer"] = None
    policy: BasePolicy = policy_class.load_from_checkpoint(**load_kwargs)
    if spec.meanflow_multistep and hasattr(policy, "set_meanflow_multistep_infer"):
        policy.set_meanflow_multistep_infer(True)
    if hasattr(policy, "set_num_k_infer"):
        policy.set_num_k_infer(int(k))
    return policy


def _build_env_runner_cfg(train_cfg: OmegaConf, *, num_episodes: int, max_episode_length: int) -> OmegaConf:
    cfg = OmegaConf.load(REPO_DIRS.ROOT / "conf" / "eval.yaml")
    with open_dict(cfg):
        cfg.env_runner.num_episodes = int(num_episodes)
        cfg.env_runner.max_episode_length = int(max_episode_length)
        cfg.env_runner.verbose = False
        cfg.env_runner.env_config.task_name = train_cfg.task_name
        cfg.env_runner.env_config.obs_mode = train_cfg.obs_mode
        cfg.env_runner.env_config.use_pc_color = train_cfg.dataset.use_pc_color
        cfg.env_runner.env_config.n_points = train_cfg.dataset.n_points
        cfg.env_runner.env_config.vis = False
        cfg.env_runner.env_config.headless = True
    return cfg


def _summarize(rows: list[dict], key: str) -> dict[str, float]:
    vals = [float(r[key]) for r in rows if key in r]
    if not vals:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "p95": 0.0}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=0)),
        "p95": float(np.percentile(arr, 95)),
    }


def _append_csv(csv_path: Path, rows: list[dict]) -> None:
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PROFILE_CSV_COLUMNS)
        if write_header:
            w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, 0.0) for k in PROFILE_CSV_COLUMNS})


def _write_report(
    all_rows: list[dict],
    summaries: list[dict],
    out_md: Path,
    *,
    seed: int,
    num_episodes: int,
    max_episode_length: int,
    old_baseline_ms: float,
) -> None:
    lines = [
        "# Rollout timing diagnostic report",
        "",
        f"- seed: `{seed}`",
        f"- num_episodes: `{num_episodes}`",
        f"- max_episode_length: `{max_episode_length}`",
        f"- device: `{'cuda' if torch.cuda.is_available() else 'cpu'}`",
        "",
        "## Summary (mean / median / std / p95)",
        "",
        "| method | K | n | mean_predict_action_ms | median_predict_action_ms | p95_predict_action_ms | "
        "mean_infer_y_ms | mean_unet_total_ms | mean_unet_ms_per_nfe | mean_env_step_ms | mean_nfe |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in summaries:
        lines.append(
            f"| {s['method']} | {s['K']} | {s['n']} | "
            f"{s['predict_action_ms']['mean']:.2f} | {s['predict_action_ms']['median']:.2f} | "
            f"{s['predict_action_ms']['p95']:.2f} | "
            f"{s['infer_y_ms']['mean']:.2f} | {s['unet_total_ms']['mean']:.2f} | "
            f"{s['unet_ms_per_nfe']['mean']:.2f} | {s['env_step_ms']['mean']:.2f} | "
            f"{s['nfe']['mean']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Old sweep reference",
            "",
            f"- FM baseline K=10 old `mean_inference_ms`: **{old_baseline_ms:.2f} ms**",
            "",
            "## Top-20 slowest actions (all methods)",
            "",
            "| method | K | episode | step | predict_action_ms | infer_y_ms | unet_total_ms | env_step_ms | nfe |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    slowest = sorted(all_rows, key=lambda r: float(r["predict_action_ms"]), reverse=True)[:20]
    for r in slowest:
        lines.append(
            f"| {r['method']} | {r['K']} | {r['episode_id']} | {r['step_id']} | "
            f"{float(r['predict_action_ms']):.2f} | {float(r.get('infer_y_ms', 0.0)):.2f} | "
            f"{float(r.get('unet_total_ms', 0.0)):.2f} | {float(r.get('env_step_ms', 0.0)):.2f} | "
            f"{float(r.get('nfe', 0.0)):.0f} |"
        )

    lines.extend(["", "## Answers", ""])
    for s in summaries:
        m = s["method"]
        lines.append(f"### {m} K={s['K']}")
        lines.append(f"- mean nfe/action: **{s['nfe']['mean']:.2f}**")
        lines.append(
            f"- mean unet_ms_per_nfe: **{s['unet_ms_per_nfe']['mean']:.2f} ms** "
            f"(unet_total {s['unet_total_ms']['mean']:.2f} ms)"
        )
        lines.append(
            f"- predict_action_ms mean/median/p95: "
            f"**{s['predict_action_ms']['mean']:.2f} / "
            f"{s['predict_action_ms']['median']:.2f} / {s['predict_action_ms']['p95']:.2f} ms**"
        )
        non_model = (
            s["predict_action_ms"]["mean"]
            - s["infer_y_ms"]["mean"]
            - s["predict_action_ms"]["mean"] * 0.0
        )
        overhead = s["predict_action_ms"]["mean"] - s["infer_y_ms"]["mean"]
        lines.append(f"- non-infer_y overhead inside predict_action: **~{overhead:.2f} ms**")
        lines.append(f"- env_step_ms mean: **{s['env_step_ms']['mean']:.2f} ms**")
        lines.append("")

    baseline = next((s for s in summaries if s["method"] == "baseline"), None)
    if baseline is not None:
        lines.append("### Baseline anomaly check")
        lines.append(
            f"- Diagnostic mean_predict_action_ms: **{baseline['predict_action_ms']['mean']:.2f} ms**"
        )
        lines.append(f"- Old sweep mean_inference_ms: **{old_baseline_ms:.2f} ms**")
        ratio = old_baseline_ms / max(baseline["predict_action_ms"]["mean"], 1e-6)
        if baseline["predict_action_ms"]["mean"] < 500:
            lines.append(
                f"- Diagnostic baseline is **~{ratio:.1f}x faster** than old sweep → "
                "old 3179 ms likely **measurement artifact or different run conditions**, "
                "not extra hidden loops in current code path."
            )
        else:
            lines.append(
                "- Diagnostic baseline remains slow; see component breakdown above for dominant bucket."
            )
        lines.append("")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_single_method(
    method_name: str,
    *,
    k: int,
    num_episodes: int,
    max_episode_length: int,
    csv_path: Path,
) -> tuple[list[dict], dict]:
    if method_name not in METHODS:
        raise SystemExit(f"Unknown method: {method_name}. Known: {sorted(METHODS)}")
    spec = METHODS[method_name]
    print(f"\n=== {method_name} K={k} ckpt={spec.ckpt_name} ===", flush=True)
    print(f"COPPELIASIM_ROOT={os.environ.get('COPPELIASIM_ROOT', '')}", flush=True)
    train_cfg = OmegaConf.load(REPO_DIRS.CKPT / spec.ckpt_name / "config.yaml")
    policy = _load_policy(spec, k=int(k))
    cfg = _build_env_runner_cfg(
        train_cfg,
        num_episodes=int(num_episodes),
        max_episode_length=int(max_episode_length),
    )
    runner = RLBenchRunner(
        **cfg.env_runner,
        profile_inference_timing=True,
        profile_method=method_name,
        profile_k=int(k),
    )
    success_list, _, _, diag = runner.run(policy, return_diagnostics=True)
    rows = diag.get("profile_rows", [])
    n_success = sum(bool(x) for x in success_list)
    print(
        f"success={n_success}/{len(success_list)} "
        f"mean_predict_action_ms={diag['mean_inference_ms']:.2f} "
        f"mean_nfe={diag['nfe_per_action']:.2f}",
        flush=True,
    )
    summary = {
        "method": method_name,
        "K": int(k),
        "n": len(rows),
        "predict_action_ms": _summarize(rows, "predict_action_ms"),
        "infer_y_ms": _summarize(rows, "infer_y_ms"),
        "unet_total_ms": _summarize(rows, "unet_total_ms"),
        "unet_ms_per_nfe": _summarize(rows, "unet_ms_per_nfe"),
        "env_step_ms": _summarize(rows, "env_step_ms"),
        "nfe": _summarize(rows, "nfe"),
    }
    _append_csv(csv_path, rows)
    return rows, summary


def _aggregate_report(
    csv_path: Path,
    out_dir: Path,
    *,
    seed: int,
    num_episodes: int,
    max_episode_length: int,
    method_names: list[str],
    k: int,
    old_baseline_ms: float,
) -> None:
    if not csv_path.exists():
        raise SystemExit(f"No CSV found: {csv_path}")
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    summaries = []
    for method_name in method_names:
        rows = [r for r in all_rows if r.get("method") == method_name]
        if not rows:
            continue
        for key in rows[0]:
            if key not in {"method", "K", "episode_id", "step_id", "success_so_far"}:
                for r in rows:
                    r[key] = float(r.get(key, 0.0) or 0.0)
        summaries.append(
            {
                "method": method_name,
                "K": int(k),
                "n": len(rows),
                "predict_action_ms": _summarize(rows, "predict_action_ms"),
                "infer_y_ms": _summarize(rows, "infer_y_ms"),
                "unet_total_ms": _summarize(rows, "unet_total_ms"),
                "unet_ms_per_nfe": _summarize(rows, "unet_ms_per_nfe"),
                "env_step_ms": _summarize(rows, "env_step_ms"),
                "nfe": _summarize(rows, "nfe"),
            }
        )
    report_path = out_dir / "rollout_timing_report.md"
    meta = {
        "seed": int(seed),
        "num_episodes": int(num_episodes),
        "max_episode_length": int(max_episode_length),
        "methods": method_names,
        "K": int(k),
        "cuda_available": bool(torch.cuda.is_available()),
        "coppelia_sim_root": os.environ.get("COPPELIASIM_ROOT"),
        "summaries": summaries,
    }
    (out_dir / "rollout_timing_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _write_report(
        all_rows,
        summaries,
        report_path,
        seed=int(seed),
        num_episodes=int(num_episodes),
        max_episode_length=int(max_episode_length),
        old_baseline_ms=float(old_baseline_ms),
    )
    print(f"\nWrote {csv_path}")
    print(f"Wrote {report_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="baseline,meanflow_multistep,shortcut")
    ap.add_argument("--method", default=None, help="Run one method in this process (fresh CoppeliaSim).")
    ap.add_argument("--aggregate-only", action="store_true", help="Build report from existing CSV.")
    ap.add_argument("--ks", type=int, default=10)
    ap.add_argument("--num-episodes", type=int, default=5)
    ap.add_argument("--max-episode-length", type=int, default=120)
    ap.add_argument("--seed", type=int, default=5678)
    ap.add_argument("--output-dir", type=Path, default=Path("results/profiling_runtime"))
    ap.add_argument("--old-baseline-ms", type=float, default=3179.22)
    args = ap.parse_args()

    _configure_runtime_env()
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "rollout_timing_diagnostic.csv"

    method_names = (
        [args.method.strip()]
        if args.method
        else [m.strip() for m in args.methods.split(",") if m.strip()]
    )

    if args.aggregate_only:
        _aggregate_report(
            csv_path,
            out_dir,
            seed=int(args.seed),
            num_episodes=int(args.num_episodes),
            max_episode_length=int(args.max_episode_length),
            method_names=method_names,
            k=int(args.ks),
            old_baseline_ms=float(args.old_baseline_ms),
        )
        return

    if args.method is None and len(method_names) > 1:
        import subprocess

        if csv_path.exists():
            csv_path.unlink()
        py = sys.executable
        script = str(Path(__file__).resolve())
        base_cmd = [
            py,
            "-u",
            script,
            "--method",
            "METHOD",
            f"--ks={int(args.ks)}",
            f"--num-episodes={int(args.num_episodes)}",
            f"--max-episode-length={int(args.max_episode_length)}",
            f"--seed={int(args.seed)}",
            f"--output-dir={out_dir}",
            f"--old-baseline-ms={float(args.old_baseline_ms)}",
        ]
        for method_name in method_names:
            cmd = [c if c != "METHOD" else method_name for c in base_cmd]
            print(f"\n>>> spawn fresh process: {method_name}", flush=True)
            subprocess.run(cmd, check=True, env=os.environ.copy())
        _aggregate_report(
            csv_path,
            out_dir,
            seed=int(args.seed),
            num_episodes=int(args.num_episodes),
            max_episode_length=int(args.max_episode_length),
            method_names=method_names,
            k=int(args.ks),
            old_baseline_ms=float(args.old_baseline_ms),
        )
        return

    set_seeds(int(args.seed))
    wandb.init(mode="disabled")
    _run_single_method(
        method_names[0],
        k=int(args.ks),
        num_episodes=int(args.num_episodes),
        max_episode_length=int(args.max_episode_length),
        csv_path=csv_path,
    )


if __name__ == "__main__":
    main()
