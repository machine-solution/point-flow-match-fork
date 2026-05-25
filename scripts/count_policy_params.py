#!/usr/bin/env python3
"""Count policy parameters and dump CSV/JSON."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import hydra
from omegaconf import OmegaConf

_diffusion_policy_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "diffusion_policy")
if os.path.exists(_diffusion_policy_path) and _diffusion_policy_path not in sys.path:
    sys.path.insert(0, _diffusion_policy_path)

from pfp import REPO_DIRS


def _count(module) -> tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return int(total), int(trainable)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-name", required=True)
    ap.add_argument("--ckpt-episode", default="latest")
    ap.add_argument("--num-k-infer", type=int, default=10)
    ap.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/efficiency/params.csv"),
    )
    ap.add_argument(
        "--output-json",
        type=Path,
        default=None,
    )
    args = ap.parse_args()

    ckpt_dir = REPO_DIRS.CKPT / args.ckpt_name
    cfg = OmegaConf.load(ckpt_dir / "config.yaml")
    policy_class = hydra.utils.get_class(cfg.model._target_)
    policy = policy_class.load_from_checkpoint(
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

    total, trainable = _count(policy)
    by_module = {}
    for name in ("obs_encoder", "diffusion_net", "phase_head", "trajectory_encoder", "trajectory_decoder"):
        if hasattr(policy, name) and getattr(policy, name) is not None:
            m_total, m_train = _count(getattr(policy, name))
            by_module[name] = {"total": m_total, "trainable": m_train}

    row = {
        "checkpoint": args.ckpt_name,
        "ckpt_episode": args.ckpt_episode,
        "total_params": total,
        "trainable_params": trainable,
        "obs_encoder_params": by_module.get("obs_encoder", {}).get("total", 0),
        "diffusion_net_params": by_module.get("diffusion_net", {}).get("total", 0),
        "phase_head_params": by_module.get("phase_head", {}).get("total", 0),
        "autoencoder_params": by_module.get("trajectory_encoder", {}).get("total", 0)
        + by_module.get("trajectory_decoder", {}).get("total", 0),
    }

    output_csv = args.output_csv.expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_csv.exists()
    with output_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(f"Wrote: {output_csv}")

    if args.output_json is not None:
        output_json = args.output_json.expanduser().resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(
                {
                    "summary": row,
                    "by_module": by_module,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Wrote: {output_json}")


if __name__ == "__main__":
    main()
