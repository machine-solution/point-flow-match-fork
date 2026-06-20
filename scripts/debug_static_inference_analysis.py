#!/usr/bin/env python3
"""Print static inference structure and parameter counts (no eval)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
_dp = REPO.parent / "diffusion_policy"
if _dp.is_dir():
    sys.path.insert(0, str(_dp))

from pfp import REPO_DIRS  # noqa: E402


def _count(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _summarize(ckpt_name: str, ckpt_episode: str = "ep1500") -> dict:
    ckpt_dir = REPO_DIRS.CKPT / ckpt_name
    cfg = OmegaConf.load(ckpt_dir / "config.yaml")
    policy_class = hydra.utils.get_class(cfg.model._target_)
    model = policy_class.load_from_checkpoint(
        ckpt_name=ckpt_name,
        ckpt_episode=ckpt_episode,
        num_k_infer=10,
        flow_schedule=getattr(cfg.model, "flow_schedule", None),
        exp_scale=getattr(cfg.model, "exp_scale", None),
        subs_factor=getattr(cfg, "subs_factor", 1),
        phase_conditioning=getattr(cfg, "phase_conditioning", None),
        phase_prediction=getattr(cfg, "phase_prediction", None),
        phase_rollout=getattr(cfg, "phase_rollout", None),
    )
    if hasattr(model, "set_meanflow_multistep_infer"):
        model.set_meanflow_multistep_infer(True)
        model.set_num_k_infer(10)

    out = {
        "ckpt_name": ckpt_name,
        "ckpt_episode": ckpt_episode,
        "policy_class": cfg.model._target_,
        "num_k_infer_ckpt": int(getattr(cfg.model, "num_k_infer", -1)),
        "n_obs_steps": int(cfg.n_obs_steps),
        "n_pred_steps": int(cfg.n_pred_steps),
        "y_dim": int(cfg.y_dim),
        "flow_schedule": str(getattr(cfg.model, "flow_schedule", None)),
        "exp_scale": float(getattr(cfg.model, "exp_scale", 0.0)),
        "dataset_n_points": int(cfg.dataset.n_points),
        "dataset_subs_factor": int(cfg.dataset.subs_factor),
        "diffusion_net_target": cfg.model.diffusion_net._target_,
        "down_dims": list(cfg.model.diffusion_net.down_dims),
        "global_cond_dim": int(cfg.model.diffusion_net.global_cond_dim),
        "diffusion_step_embed_dim": int(cfg.model.diffusion_net.diffusion_step_embed_dim),
        "params_total": _count(model),
        "params_obs_encoder": _count(model.obs_encoder),
        "params_diffusion_net": _count(model.diffusion_net),
    }
    if hasattr(model, "interval_mlp"):
        out["params_interval_mlp"] = _count(model.interval_mlp)
    if hasattr(model, "d_embed"):
        out["params_d_embed"] = _count(model.d_embed)
    if hasattr(model, "meanflow_multistep_infer"):
        out["meanflow_multistep_infer"] = bool(model.meanflow_multistep_infer)
        out["sampler_mode"] = str(getattr(model, "sampler_mode", ""))
    pt = ckpt_dir / f"{ckpt_episode}-rank0.pt"
    if not pt.exists():
        pt = ckpt_dir / "latest-rank0.pt"
    out["checkpoint_file_mb"] = round(pt.stat().st_size / (1024 * 1024), 1) if pt.exists() else None
    return out


def main() -> None:
    specs = [
        ("1779122560-baseline-many-ckpts", "ep1500"),
        ("meanflow_open_fridge_1365", "latest"),
        ("shortcut_open_fridge_1385", "latest"),
    ]
    rows = [_summarize(name, ep) for name, ep in specs]
    out_path = REPO / "results" / "profiling_static" / "model_static_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
