#!/usr/bin/env python3
"""Compare FM baseline vs correct multi-step MeanFlow across K values."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import wandb

_diffusion_policy_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "diffusion_policy")
if os.path.exists(_diffusion_policy_path) and _diffusion_policy_path not in sys.path:
    sys.path.insert(0, _diffusion_policy_path)

import torch.cuda.amp.grad_scaler as _cuda_gs

if not hasattr(_cuda_gs, "_refresh_per_optimizer_state"):
    try:
        from torch.amp.grad_scaler import _refresh_per_optimizer_state

        _cuda_gs._refresh_per_optimizer_state = _refresh_per_optimizer_state
    except ImportError:
        pass

from pfp import REPO_DIRS


@dataclass(frozen=True)
class MethodSpec:
    name: str
    ckpt_name: str
    ckpt_episode: str
    meanflow_multistep: bool = False
    meanflow_schedule: str = "uniform"


METHOD_ALIASES = {
    "baseline": "baseline",
    "fm": "baseline",
    "fm_baseline": "baseline",
    "meanflow": "meanflow_multistep",
    "meanflow_multistep": "meanflow_multistep",
}


def _parse_list(s: str, *, cast=str) -> list:
    out = []
    for x in s.split(","):
        x = x.strip()
        if x:
            out.append(cast(x))
    if not out:
        raise ValueError("empty list")
    return out


def _configure_runtime_env() -> None:
    cs41 = REPO_DIRS.ROOT / "CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
    if cs41.is_dir():
        os.environ["COPPELIASIM_ROOT"] = str(cs41)
        os.environ["LD_LIBRARY_PATH"] = f"{cs41}:{os.environ.get('LD_LIBRARY_PATH', '')}"
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(cs41)
    dp = (REPO_DIRS.ROOT.parent / "diffusion_policy").resolve()
    if dp.is_dir():
        os.environ["PYTHONPATH"] = f"{dp}:{REPO_DIRS.ROOT}:{os.environ.get('PYTHONPATH', '')}"
    os.environ.setdefault("PYTHONUNBUFFERED", "1")


def _resolve_methods(
    names: list[str],
    *,
    baseline_ckpt_name: str,
    baseline_ckpt_episode: str,
    meanflow_ckpt_name: str,
    meanflow_ckpt_episode: str,
) -> list[MethodSpec]:
    specs: list[MethodSpec] = []
    for raw in names:
        key = METHOD_ALIASES.get(raw.strip().lower(), raw.strip().lower())
        if key == "baseline":
            specs.append(
                MethodSpec(
                    name="baseline",
                    ckpt_name=baseline_ckpt_name,
                    ckpt_episode=baseline_ckpt_episode,
                )
            )
        elif key == "meanflow_multistep":
            specs.append(
                MethodSpec(
                    name="meanflow_multistep",
                    ckpt_name=meanflow_ckpt_name,
                    ckpt_episode=meanflow_ckpt_episode,
                    meanflow_multistep=True,
                    meanflow_schedule="uniform",
                )
            )
        else:
            raise ValueError(f"Unknown method {raw!r}. Expected baseline or meanflow_multistep.")
    return specs


def _write_markdown(out_md: Path, *, details: dict, rows: list[dict]) -> None:
    lines = [
        "# FM baseline vs MeanFlow multistep K-sweep",
        "",
        f"- seed: `{details['seed']}`",
        f"- num_episodes: `{details['num_episodes']}`",
        f"- max_episode_length: `{details['max_episode_length']}`",
        f"- baseline checkpoint: `{details['baseline_ckpt_name']}` ({details['baseline_ckpt_episode']})",
        f"- meanflow checkpoint: `{details['meanflow_ckpt_name']}` ({details['meanflow_ckpt_episode']})",
        "",
        "| method | K | success | success_rate | nfe/action | mean_inference_ms |",
        "| ------ | - | ------- | ------------ | ---------- | ----------------- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['method']} | {r['k']} | {r['success']} | {float(r['success_rate']):.4f} | "
            f"{float(r['nfe_per_action']):.1f} | {float(r['mean_inference_ms']):.2f} |"
        )

    by_method: dict[str, list[dict]] = {}
    for r in rows:
        by_method.setdefault(str(r["method"]), []).append(r)

    lines.extend(["", "## Summary", ""])
    best_lines: list[str] = []
    for method, mrows in sorted(by_method.items()):
        best = max(mrows, key=lambda x: float(x["success_rate"]))
        best_lines.append(
            f"- Best {method} K={best['k']}: success_rate={float(best['success_rate']):.4f} "
            f"({best['success']}/{best['num_episodes']})"
        )
    lines.extend(best_lines)

    ks = sorted({int(r["k"]) for r in rows})
    base_map = {int(r["k"]): r for r in rows if r["method"] == "baseline"}
    mf_map = {int(r["k"]): r for r in rows if r["method"] == "meanflow_multistep"}
    if base_map and mf_map:
        lines.extend(["", "## MeanFlow vs baseline at same K", ""])
        for k in ks:
            b = base_map.get(k)
            m = mf_map.get(k)
            if b is None or m is None:
                continue
            lines.append(
                f"- K={k}: baseline={float(b['success_rate']):.4f} "
                f"meanflow={float(m['success_rate']):.4f} "
                f"(delta={float(m['success_rate']) - float(b['success_rate']):+.4f})"
            )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="baseline,meanflow_multistep")
    ap.add_argument("--ks", default="1,2,5,8,10,15")
    ap.add_argument("--num-episodes", type=int, default=100)
    ap.add_argument("--max-episode-length", type=int, default=120)
    ap.add_argument("--seed", type=int, default=5678)
    ap.add_argument("--baseline-ckpt-name", default="1779122560-baseline-many-ckpts")
    ap.add_argument("--baseline-ckpt-episode", default="ep1500")
    ap.add_argument("--meanflow-ckpt-name", default="meanflow_open_fridge_1365")
    ap.add_argument("--meanflow-ckpt-episode", default="latest")
    ap.add_argument("--output-dir", type=Path, default=Path("results/efficiency"))
    ap.add_argument("--resume-existing", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Quick run: num_episodes=2, ks=1,10.",
    )
    args = ap.parse_args()

    if args.smoke:
        args.num_episodes = 2
        args.ks = "1,10"

    methods = _resolve_methods(
        _parse_list(args.methods),
        baseline_ckpt_name=str(args.baseline_ckpt_name),
        baseline_ckpt_episode=str(args.baseline_ckpt_episode),
        meanflow_ckpt_name=str(args.meanflow_ckpt_name),
        meanflow_ckpt_episode=str(args.meanflow_ckpt_episode),
    )
    ks = _parse_list(args.ks, cast=int)

    for spec in methods:
        ckpt_dir = REPO_DIRS.CKPT / spec.ckpt_name
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint not found for {spec.name}: {ckpt_dir}")

    _configure_runtime_env()
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "fm_vs_meanflow_k_sweep.csv"
    out_json = out_dir / "fm_vs_meanflow_k_sweep.json"
    out_md = out_dir / "fm_vs_meanflow_k_sweep.md"

    fieldnames = [
        "method",
        "k",
        "success",
        "success_rate",
        "num_episodes",
        "mean_inference_ms",
        "std_inference_ms",
        "nfe_per_action",
        "sampler_mode",
        "checkpoint",
        "ckpt_episode",
        "seed",
        "flow_schedule",
        "exp_scale",
        "meanflow_multistep_infer",
        "meanflow_schedule",
    ]

    rows: list[dict] = []
    details = {
        "seed": int(args.seed),
        "num_episodes": int(args.num_episodes),
        "max_episode_length": int(args.max_episode_length),
        "ks": ks,
        "methods": [m.name for m in methods],
        "baseline_ckpt_name": str(args.baseline_ckpt_name),
        "baseline_ckpt_episode": str(args.baseline_ckpt_episode),
        "meanflow_ckpt_name": str(args.meanflow_ckpt_name),
        "meanflow_ckpt_episode": str(args.meanflow_ckpt_episode),
        "smoke": bool(args.smoke),
        "runs": [],
    }
    done: set[tuple[str, int]] = set()
    if args.resume_existing and out_csv.exists():
        with out_csv.open("r", newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append(r)
                done.add((str(r["method"]), int(r["k"])))
        print(f"Resuming from {out_csv} (done pairs={len(done)})")

    def _flush() -> None:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        out_json.write_text(json.dumps(details, indent=2), encoding="utf-8")
        _write_markdown(out_md, details=details, rows=rows)

    wandb.init(mode="disabled")
    script_path = (REPO_DIRS.ROOT / "scripts" / "validate_accuracy.py").resolve()
    failed: list[str] = []

    for spec in methods:
        for k in ks:
            key = (spec.name, int(k))
            if key in done:
                print(f"Skip {spec.name} K={k}: already done")
                continue
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            tmp_json = out_dir / f".fm_vs_mf_{spec.name}_k{k}_tmp.json"
            cmd = [
                sys.executable,
                str(script_path),
                f"policy.ckpt_name={spec.ckpt_name}",
                f"policy.ckpt_episode={spec.ckpt_episode}",
                f"policy.num_k_infer={int(k)}",
                f"env_runner.num_episodes={int(args.num_episodes)}",
                f"env_runner.max_episode_length={int(args.max_episode_length)}",
                f"env_runner.verbose={str(bool(args.verbose))}",
                f"seed={int(args.seed)}",
                f"results_json={tmp_json}",
            ]
            if spec.meanflow_multistep:
                cmd.append("+policy.meanflow_multistep_infer=true")
                cmd.append(f"+policy.meanflow_schedule={spec.meanflow_schedule}")
            print(f"=== Eval method={spec.name} K={k} episodes={args.num_episodes} ===")
            proc = subprocess.run(cmd, cwd=str(REPO_DIRS.ROOT), env=os.environ.copy(), capture_output=True, text=True)
            if proc.returncode != 0:
                print(proc.stdout[-4000:], file=sys.stderr)
                print(proc.stderr[-4000:], file=sys.stderr)
                failed.append(f"{spec.name}:K={k}")
                continue
            if not tmp_json.exists():
                failed.append(f"{spec.name}:K={k}")
                continue
            run = json.loads(tmp_json.read_text(encoding="utf-8"))
            try:
                tmp_json.unlink()
            except OSError:
                pass

            n_success = int(run["num_success"])
            n = int(run["num_episodes"])
            success_rate = float(run["accuracy"])
            nfe = float(run.get("nfe_per_action", 0.0))
            if abs(nfe - float(k)) > 1e-3:
                raise RuntimeError(f"{spec.name} K={k}: expected nfe/action={k}, got {nfe}")
            if spec.meanflow_multistep and not bool(run.get("meanflow_multistep_infer", False)):
                raise RuntimeError(f"{spec.name} K={k}: meanflow_multistep_infer is false")

            row = {
                "method": spec.name,
                "k": str(k),
                "success": str(n_success),
                "success_rate": f"{success_rate:.6f}",
                "num_episodes": str(n),
                "mean_inference_ms": f"{float(run.get('mean_inference_ms', 0.0)):.4f}",
                "std_inference_ms": f"{float(run.get('std_inference_ms', 0.0)):.4f}",
                "nfe_per_action": f"{nfe:.4f}",
                "sampler_mode": str(run.get("sampler_mode", "unknown")),
                "checkpoint": spec.ckpt_name,
                "ckpt_episode": spec.ckpt_episode,
                "seed": str(int(args.seed)),
                "flow_schedule": str(run.get("flow_schedule", "")),
                "exp_scale": str(run.get("exp_scale", "")),
                "meanflow_multistep_infer": str(bool(run.get("meanflow_multistep_infer", False))),
                "meanflow_schedule": str(run.get("meanflow_schedule", "")),
            }
            rows.append(row)
            details["runs"].append({**row, "success_rate": success_rate, "nfe_per_action": nfe})
            _flush()
            print(
                f"{spec.name} K={k} | success={n_success}/{n} ({100*success_rate:.1f}%) | "
                f"nfe/action={nfe:.1f} | infer={float(row['mean_inference_ms']):.2f} ms"
            )

    _flush()
    print(f"Wrote CSV:  {out_csv}")
    print(f"Wrote JSON: {out_json}")
    print(f"Wrote MD:   {out_md}")
    wandb.finish()
    if failed:
        raise SystemExit(f"Failed runs: {failed}")


if __name__ == "__main__":
    main()
