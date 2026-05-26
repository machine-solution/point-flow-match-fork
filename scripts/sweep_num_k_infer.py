#!/usr/bin/env python3
"""Evaluate one checkpoint across multiple num_k_infer values."""
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

# PyTorch 2.7 compatibility for Composer imports in checkpoints.
import torch.cuda.amp.grad_scaler as _cuda_gs

if not hasattr(_cuda_gs, "_refresh_per_optimizer_state"):
    try:
        from torch.amp.grad_scaler import _refresh_per_optimizer_state

        _cuda_gs._refresh_per_optimizer_state = _refresh_per_optimizer_state
    except ImportError:
        pass

from pfp import REPO_DIRS


def _parse_ks(s: str) -> list[int]:
    out = []
    for x in s.split(","):
        x = x.strip()
        if x:
            out.append(int(x))
    if not out:
        raise ValueError("num_k_infer sweep list is empty")
    return out


def _to_jsonable(v):
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    if isinstance(v, dict):
        return {str(k): _to_jsonable(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_to_jsonable(x) for x in v]
    return str(v)


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-name", default=None, help="Checkpoint run name under ckpt/.")
    ap.add_argument("--checkpoint", default=None, help="Alias for --ckpt-name.")
    ap.add_argument("--ckpt-episode", default="ep1500")
    ap.add_argument("--num-episodes", type=int, default=50)
    ap.add_argument("--max-episode-length", type=int, default=120)
    ap.add_argument("--seed", type=int, default=5678)
    ap.add_argument("--ks", default="1,2,4,6,8,10")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--flow-schedule", default=None)
    ap.add_argument("--exp-scale", type=float, default=None)
    ap.add_argument("--subs-factor", type=int, default=None)
    ap.add_argument("--phase-conditioning", choices=("enabled", "disabled"), default="disabled")
    ap.add_argument("--phase-prediction", choices=("enabled", "disabled"), default="disabled")
    ap.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/efficiency/k_sweep.csv"),
        help="Output CSV path (default: results/efficiency/k_sweep.csv).",
    )
    ap.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional detailed JSON dump with per-k diagnostics.",
    )
    ap.add_argument(
        "--resume-existing",
        action="store_true",
        help="If output CSV exists, load rows and skip already completed K values.",
    )
    args = ap.parse_args()
    ckpt_name = args.ckpt_name or args.checkpoint
    if not ckpt_name:
        raise SystemExit("Provide --ckpt-name <name> (or --checkpoint <name>).")

    ks = _parse_ks(args.ks)
    ckpt_dir = REPO_DIRS.CKPT / ckpt_name
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    _configure_runtime_env()
    output_csv = args.output_csv.expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    details = {
        "checkpoint": ckpt_name,
        "checkpoint_episode": args.ckpt_episode,
        "seed": int(args.seed),
        "num_episodes": int(args.num_episodes),
        "max_episode_length": int(args.max_episode_length),
        "ks": ks,
        "runs": [],
    }
    done_k: set[int] = set()
    if args.resume_existing and output_csv.exists():
        with output_csv.open("r", newline="", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                k = int(r["num_k_infer"])
                done_k.add(k)
                rows.append(
                    {
                        "checkpoint": r["checkpoint"],
                        "ckpt_episode": r.get("ckpt_episode", args.ckpt_episode),
                        "num_k_infer": k,
                        "num_episodes": int(r["num_episodes"]),
                        "success_rate": float(r["success_rate"]),
                        "mean_episode_reward_or_success": float(r["mean_episode_reward_or_success"]),
                        "mean_inference_ms": float(r["mean_inference_ms"]),
                        "std_inference_ms": float(r["std_inference_ms"]),
                        "nfe_per_action": float(r["nfe_per_action"]),
                        "mean_episode_time_s": float(r.get("mean_episode_time_s", 0.0)),
                        "max_gpu_memory_mb": float(r["max_gpu_memory_mb"])
                        if r.get("max_gpu_memory_mb")
                        else None,
                    }
                )
        print(f"Resuming from existing CSV: {output_csv} (done K={sorted(done_k)})")

    wandb.init(mode="disabled")
    fieldnames = [
        "checkpoint",
        "ckpt_episode",
        "num_k_infer",
        "num_episodes",
        "success_rate",
        "mean_episode_reward_or_success",
        "mean_inference_ms",
        "std_inference_ms",
        "nfe_per_action",
        "mean_episode_time_s",
        "max_gpu_memory_mb",
    ]

    def _flush_partial() -> None:
        with output_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        if args.output_json is not None:
            out_json = args.output_json.expanduser().resolve()
            out_json.parent.mkdir(parents=True, exist_ok=True)
            out_json.write_text(json.dumps(details, indent=2), encoding="utf-8")

    phase_conditioning = args.phase_conditioning
    phase_prediction = args.phase_prediction
    script_path = (REPO_DIRS.ROOT / "scripts" / "validate_accuracy.py").resolve()
    failed_ks: list[int] = []
    for k in ks:
        if k in done_k:
            print(f"Skip K={k}: already present in {output_csv}")
            continue
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        tmp_json = output_csv.parent / f".k{k}_tmp_result.json"
        cmd = [
            sys.executable,
            str(script_path),
            f"policy.ckpt_name={ckpt_name}",
            f"policy.ckpt_episode={args.ckpt_episode}",
            f"policy.num_k_infer={int(k)}",
            f"env_runner.num_episodes={int(args.num_episodes)}",
            f"env_runner.max_episode_length={int(args.max_episode_length)}",
            f"env_runner.verbose={str(bool(args.verbose))}",
            f"seed={int(args.seed)}",
            f"phase_conditioning={phase_conditioning}",
            f"phase_prediction={phase_prediction}",
            f"results_json={tmp_json}",
        ]
        if args.flow_schedule is not None:
            cmd.append(f"policy.flow_schedule={args.flow_schedule}")
        if args.exp_scale is not None:
            cmd.append(f"policy.exp_scale={float(args.exp_scale)}")
        if args.subs_factor is not None:
            cmd.append(f"policy.subs_factor={int(args.subs_factor)}")
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_DIRS.ROOT),
            env=os.environ.copy(),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            print(f"K={k} failed with exit={proc.returncode}", file=sys.stderr)
            print(proc.stdout[-4000:], file=sys.stderr)
            print(proc.stderr[-4000:], file=sys.stderr)
            failed_ks.append(int(k))
            continue
        if not tmp_json.exists():
            print(f"K={k} failed: missing result JSON {tmp_json}", file=sys.stderr)
            failed_ks.append(int(k))
            continue
        run = json.loads(tmp_json.read_text(encoding="utf-8"))
        n = int(run["num_episodes"])
        n_success = int(run["num_success"])
        success_rate = float(run["accuracy"])
        diagnostics = {
            "mean_inference_ms": float(run.get("mean_inference_ms", 0.0)),
            "std_inference_ms": float(run.get("std_inference_ms", 0.0)),
            "nfe_per_action": float(run.get("nfe_per_action", 0.0)),
            "mean_episode_time_s": float(run.get("mean_episode_time_s", 0.0)),
            "policy_inference_diagnostics": run.get("policy_inference_diagnostics", {}),
        }
        max_gpu_memory_mb = None
        try:
            tmp_json.unlink()
        except OSError:
            pass
        row = {
            "checkpoint": ckpt_name,
            "ckpt_episode": args.ckpt_episode,
            "num_k_infer": int(k),
            "num_episodes": int(n),
            "success_rate": success_rate,
            "mean_episode_reward_or_success": success_rate,
            "mean_inference_ms": float(diagnostics["mean_inference_ms"]),
            "std_inference_ms": float(diagnostics["std_inference_ms"]),
            "nfe_per_action": float(diagnostics["nfe_per_action"]),
            "mean_episode_time_s": float(diagnostics["mean_episode_time_s"]),
            "max_gpu_memory_mb": max_gpu_memory_mb,
        }
        rows.append(row)
        details["runs"].append(
            {
                **row,
                "num_success": n_success,
                "policy_inference_diagnostics": _to_jsonable(diagnostics.get("policy_inference_diagnostics", {})),
            }
        )
        _flush_partial()
        done_k.add(int(k))
        print(
            f"K={k:>2d} | success={n_success}/{n} ({100*success_rate:.1f}%) | "
            f"infer={row['mean_inference_ms']:.2f}±{row['std_inference_ms']:.2f} ms | "
            f"nfe/action={row['nfe_per_action']:.2f} | episode_time={row['mean_episode_time_s']:.2f}s"
        )

    _flush_partial()
    print(f"Wrote CSV: {output_csv}")

    if args.output_json is not None:
        out_json = args.output_json.expanduser().resolve()
        print(f"Wrote JSON: {out_json}")

    wandb.finish()
    if failed_ks:
        raise SystemExit(f"K-sweep failed for K values: {failed_ks}")


if __name__ == "__main__":
    main()
