#!/usr/bin/env python3
"""Sweep MeanFlow multi-step inference time schedules at fixed K."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
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
from pfp.common.meanflow_utils import (
    MEANFLOW_INFER_SCHEDULES,
    build_meanflow_time_grid,
    meanflow_grid_payload,
)


def _parse_schedules(s: str) -> list[str]:
    out = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        if x not in MEANFLOW_INFER_SCHEDULES:
            raise ValueError(f"Unknown schedule {x!r}. Expected one of {MEANFLOW_INFER_SCHEDULES}")
        out.append(x)
    if not out:
        raise ValueError("schedule list is empty")
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


def _grid_sanity(k: int, schedule: str, *, exp_scale: float) -> dict:
    grid = build_meanflow_time_grid(k, schedule, exp_scale=exp_scale)
    payload = meanflow_grid_payload(grid)
    assert len(payload["time_grid"]) == k + 1
    assert abs(payload["time_grid"][0]) < 1e-6
    assert abs(payload["time_grid"][-1] - 1.0) < 1e-5
    assert all(d > 0 for d in payload["dt_grid"])
    assert all(
        payload["time_grid"][i + 1] > payload["time_grid"][i] for i in range(len(payload["dt_grid"]))
    )
    return payload


def _write_markdown(out_md: Path, *, details: dict, rows: list[dict]) -> None:
    lines = [
        "# MeanFlow schedule sweep",
        "",
        f"- checkpoint: `{details['checkpoint']}` ({details['checkpoint_episode']})",
        f"- K: `{details['k']}`",
        f"- num_episodes: `{details['num_episodes']}`",
        f"- seed: `{details['seed']}`",
        "",
        "| schedule | K | success | success_rate | nfe/action | mean_inference_ms | time_grid | dt_grid |",
        "| -------- | - | ------- | ------------ | ---------- | ----------------- | --------- | ------- |",
    ]
    for r in rows:
        tg = json.dumps(r.get("time_grid", []), ensure_ascii=False)
        dg = json.dumps(r.get("dt_grid", []), ensure_ascii=False)
        lines.append(
            f"| {r['schedule']} | {r['k']} | {r['success']} | {r['success_rate']:.4f} | "
            f"{r['nfe_per_action']:.1f} | {r['mean_inference_ms']:.2f} | `{tg}` | `{dg}` |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-name", required=True)
    ap.add_argument("--ckpt-episode", default="latest")
    ap.add_argument("--num-episodes", type=int, default=100)
    ap.add_argument("--max-episode-length", type=int, default=120)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument(
        "--schedules",
        default="uniform,fm_exp,reverse_exp,cosine,beta_2_2,beta_3_3",
    )
    ap.add_argument("--seed", type=int, default=5678)
    ap.add_argument("--exp-scale", type=float, default=4.0)
    ap.add_argument("--output-dir", type=Path, default=Path("results/efficiency"))
    ap.add_argument("--resume-existing", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    schedules = _parse_schedules(args.schedules)
    k = int(args.k)
    ckpt_name = str(args.ckpt_name)
    ckpt_dir = REPO_DIRS.CKPT / ckpt_name
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    _configure_runtime_env()
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"meanflow_schedule_sweep_k{k}.csv"
    out_json = out_dir / f"meanflow_schedule_sweep_k{k}.json"
    out_md = out_dir / f"meanflow_schedule_sweep_k{k}.md"

    print("=== Grid sanity checks ===")
    grid_by_schedule: dict[str, dict] = {}
    for sched in schedules:
        grid_by_schedule[sched] = _grid_sanity(k, sched, exp_scale=float(args.exp_scale))
        print(
            f"schedule={sched}: dt_max_idx={grid_by_schedule[sched]['dt_max_idx']} "
            f"dt={[round(x, 4) for x in grid_by_schedule[sched]['dt_grid']]}"
        )

    rows: list[dict] = []
    details: dict = {
        "checkpoint": ckpt_name,
        "checkpoint_episode": args.ckpt_episode,
        "seed": int(args.seed),
        "num_episodes": int(args.num_episodes),
        "max_episode_length": int(args.max_episode_length),
        "k": k,
        "exp_scale": float(args.exp_scale),
        "schedules": schedules,
        "grids": grid_by_schedule,
        "runs": [],
    }
    done: set[str] = set()
    if args.resume_existing and out_csv.exists():
        with out_csv.open("r", newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append(r)
                done.add(str(r["schedule"]))
        print(f"Resuming from {out_csv} (done schedules={sorted(done)})")

    fieldnames = [
        "checkpoint",
        "ckpt_episode",
        "schedule",
        "k",
        "num_episodes",
        "success",
        "success_rate",
        "mean_inference_ms",
        "std_inference_ms",
        "nfe_per_action",
        "mean_episode_time_s",
        "sampler_mode",
    ]

    def _flush() -> None:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        out_json.write_text(json.dumps(details, indent=2), encoding="utf-8")
        _write_markdown(out_md, details=details, rows=details["runs"])

    wandb.init(mode="disabled")
    script_path = (REPO_DIRS.ROOT / "scripts" / "validate_accuracy.py").resolve()
    failed: list[str] = []

    for sched in schedules:
        if sched in done:
            print(f"Skip schedule={sched}: already in CSV")
            continue
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        tmp_json = out_dir / f".schedule_{sched}_tmp.json"
        cmd = [
            sys.executable,
            str(script_path),
            f"policy.ckpt_name={ckpt_name}",
            f"policy.ckpt_episode={args.ckpt_episode}",
            f"policy.num_k_infer={k}",
            "+policy.meanflow_multistep_infer=true",
            f"+policy.meanflow_schedule={sched}",
            f"env_runner.num_episodes={int(args.num_episodes)}",
            f"env_runner.max_episode_length={int(args.max_episode_length)}",
            f"env_runner.verbose={str(bool(args.verbose))}",
            f"seed={int(args.seed)}",
            f"results_json={tmp_json}",
        ]
        print(f"=== Eval schedule={sched} K={k} episodes={args.num_episodes} ===")
        proc = subprocess.run(cmd, cwd=str(REPO_DIRS.ROOT), env=os.environ.copy(), capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stdout[-4000:], file=sys.stderr)
            print(proc.stderr[-4000:], file=sys.stderr)
            failed.append(sched)
            continue
        if not tmp_json.exists():
            print(f"Missing result JSON for schedule={sched}", file=sys.stderr)
            failed.append(sched)
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
            raise RuntimeError(f"schedule={sched}: expected nfe/action={k}, got {nfe}")

        grid_payload = grid_by_schedule[sched]
        row = {
            "checkpoint": ckpt_name,
            "ckpt_episode": str(args.ckpt_episode),
            "schedule": sched,
            "k": str(k),
            "num_episodes": str(n),
            "success": str(n_success),
            "success_rate": f"{success_rate:.6f}",
            "mean_inference_ms": f"{float(run.get('mean_inference_ms', 0.0)):.4f}",
            "std_inference_ms": f"{float(run.get('std_inference_ms', 0.0)):.4f}",
            "nfe_per_action": f"{nfe:.4f}",
            "mean_episode_time_s": f"{float(run.get('mean_episode_time_s', 0.0)):.4f}",
            "sampler_mode": str(run.get("sampler_mode", "unknown")),
        }
        rows.append(row)
        details["runs"].append(
            {
                **row,
                "success_rate": success_rate,
                "mean_inference_ms": float(row["mean_inference_ms"]),
                "std_inference_ms": float(row["std_inference_ms"]),
                "nfe_per_action": nfe,
                "mean_episode_time_s": float(row["mean_episode_time_s"]),
                "time_grid": grid_payload["time_grid"],
                "dt_grid": grid_payload["dt_grid"],
                "dt_max_idx": grid_payload["dt_max_idx"],
                "meanflow_schedule": str(run.get("meanflow_schedule", sched)),
            }
        )
        _flush()
        print(
            f"schedule={sched} | success={n_success}/{n} ({100*success_rate:.1f}%) | "
            f"nfe/action={nfe:.1f} | infer={float(row['mean_inference_ms']):.2f} ms"
        )

    _flush()
    print(f"Wrote CSV:  {out_csv}")
    print(f"Wrote JSON: {out_json}")
    print(f"Wrote MD:   {out_md}")
    wandb.finish()
    if failed:
        raise SystemExit(f"Failed schedules: {failed}")


if __name__ == "__main__":
    main()
