#!/usr/bin/env python3
"""Micro-benchmark policy inference latency and NFE."""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

_diffusion_policy_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "diffusion_policy")
if os.path.exists(_diffusion_policy_path) and _diffusion_policy_path not in sys.path:
    sys.path.insert(0, _diffusion_policy_path)

from pfp import DEVICE, REPO_DIRS
from pfp.policy.base_policy import BasePolicy


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _summary(vals: list[float]) -> dict[str, float]:
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(arr.mean()) if arr.size else 0.0,
        "std": float(arr.std(ddof=0)) if arr.size else 0.0,
        "p50": float(np.percentile(arr, 50)) if arr.size else 0.0,
        "p90": float(np.percentile(arr, 90)) if arr.size else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-name", required=True)
    ap.add_argument("--ckpt-episode", default="latest")
    ap.add_argument("--num-k-infer", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--warmup-iters", type=int, default=20)
    ap.add_argument("--timed-iters", type=int, default=100)
    ap.add_argument("--output-csv", type=Path, default=Path("results/efficiency/latency.csv"))
    args = ap.parse_args()

    ckpt_dir = REPO_DIRS.CKPT / args.ckpt_name
    cfg = OmegaConf.load(ckpt_dir / "config.yaml")
    policy_class = hydra.utils.get_class(cfg.model._target_)
    policy: BasePolicy = policy_class.load_from_checkpoint(
        ckpt_name=args.ckpt_name,
        ckpt_episode=args.ckpt_episode,
        num_k_infer=args.num_k_infer,
        flow_schedule=getattr(cfg.model, "flow_schedule", None),
        exp_scale=getattr(cfg.model, "exp_scale", None),
        subs_factor=getattr(cfg.model, "subs_factor", 1),
        phase_conditioning=getattr(cfg, "phase_conditioning", None),
        phase_prediction=getattr(cfg, "phase_prediction", None),
        phase_rollout=getattr(cfg, "phase_rollout", None),
    )
    print(f"[model] class={policy.__class__.__name__} target={cfg.model._target_}")
    print(f"[model] num_k_infer={getattr(policy, 'num_k_infer', None)}")
    if hasattr(policy, "meanflow_enabled"):
        print(
            f"[model] meanflow.enabled={getattr(policy, 'meanflow_enabled', None)} "
            f"one_step={getattr(policy, 'meanflow_one_step', None)} "
            f"interval_embed_dim={getattr(policy, 'interval_embed_dim', None)}"
        )
    if hasattr(policy, "reset_inference_diagnostics"):
        policy.reset_inference_diagnostics()

    B = int(args.batch_size)
    n_obs_steps = int(cfg.n_obs_steps)
    n_points = int(cfg.dataset.n_points)
    y_dim = int(cfg.y_dim)
    pcd = torch.randn((B, n_obs_steps, n_points, 3), device=DEVICE)
    robot_state_obs = torch.randn((B, n_obs_steps, y_dim), device=DEVICE)

    def _call_once() -> float:
        _sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = policy.infer_y(pcd, robot_state_obs)
        _sync()
        return (time.perf_counter() - t0) * 1000.0

    for _ in range(args.warmup_iters):
        _call_once()
    times_ms = [_call_once() for _ in range(args.timed_iters)]
    stats = _summary(times_ms)
    nfe_values = [float(getattr(policy, "last_infer_nfe", args.num_k_infer)) for _ in range(args.timed_iters)]

    row = {
        "checkpoint": args.ckpt_name,
        "ckpt_episode": args.ckpt_episode,
        "num_k_infer": int(args.num_k_infer),
        "batch_size": B,
        "timed_iters": int(args.timed_iters),
        "mean_ms": stats["mean"],
        "std_ms": stats["std"],
        "p50_ms": stats["p50"],
        "p90_ms": stats["p90"],
        "nfe_per_action": float(statistics.fmean(nfe_values)) if nfe_values else 0.0,
    }

    out_csv = args.output_csv.expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_csv.exists()
    with out_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(f"Wrote: {out_csv}")
    print(
        f"Latency: mean={row['mean_ms']:.2f} ms std={row['std_ms']:.2f} "
        f"p50={row['p50_ms']:.2f} p90={row['p90_ms']:.2f} NFE/action={row['nfe_per_action']:.2f}"
    )


if __name__ == "__main__":
    main()
