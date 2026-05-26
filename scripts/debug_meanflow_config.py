#!/usr/bin/env python3
"""Debug Hydra composition/instantiation for MeanFlow config."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import hydra
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

_diffusion_policy_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "diffusion_policy")
if os.path.exists(_diffusion_policy_path) and _diffusion_policy_path not in sys.path:
    sys.path.insert(0, _diffusion_policy_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="train")
    ap.add_argument(
        "--overrides",
        nargs="*",
        default=["+experiment=pointflowmatch_meanflow"],
    )
    args = ap.parse_args()

    conf_dir = (Path(__file__).resolve().parents[1] / "conf").resolve()
    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg = compose(config_name=args.config_name, overrides=args.overrides)
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    print("===== Resolved Config =====")
    print(OmegaConf.to_yaml(cfg))

    model = hydra.utils.instantiate(
        cfg.model,
        phase_conditioning=getattr(cfg, "phase_conditioning", None),
        phase_prediction=getattr(cfg, "phase_prediction", None),
        phase_rollout=getattr(cfg, "phase_rollout", None),
    )
    print("===== Model =====")
    print(f"class={model.__class__.__name__}")
    print(f"target={cfg.model._target_}")
    print(f"num_k_infer={getattr(model, 'num_k_infer', None)}")
    print(f"interval_embed_dim={getattr(model, 'interval_embed_dim', None)}")
    print(f"meanflow_enabled={getattr(model, 'meanflow_enabled', None)}")
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params_total={n_params:,} trainable={n_train:,}")

    # Dummy train-path loss call (no environment dependencies).
    B = 2
    pcd = torch.randn(B, cfg.n_obs_steps, cfg.dataset.n_points, 3)
    robot_state_obs = torch.randn(B, cfg.n_obs_steps, cfg.y_dim)
    robot_state_pred = torch.randn(B, cfg.n_pred_steps, cfg.y_dim)
    print("===== Dummy Shapes =====")
    print(f"pcd={tuple(pcd.shape)} robot_state_obs={tuple(robot_state_obs.shape)} robot_state_pred={tuple(robot_state_pred.shape)}")
    with torch.no_grad():
        loss_xyz, loss_rot6d, loss_grip, _, _ = model.calculate_loss(
            pcd, robot_state_obs, robot_state_pred, phase=None
        )
    print("===== Dummy Loss =====")
    print(
        f"loss_xyz={float(loss_xyz):.6f} "
        f"loss_rot6d={float(loss_rot6d):.6f} "
        f"loss_grip={float(loss_grip):.6f}"
    )

    # Dummy infer-path call and explicit one-step verification.
    with torch.no_grad():
        _ = model.infer_y(pcd, robot_state_obs)
    print("===== Dummy Infer =====")
    print(f"last_infer_nfe={getattr(model, 'last_infer_nfe', None)}")
    print(f"meanflow_nfe={getattr(model, 'meanflow_nfe', None)}")
    if getattr(model, "last_infer_nfe", None) != 1:
        raise RuntimeError("MeanFlow infer_y must execute exactly one diffusion forward call.")
    if getattr(model, "meanflow_nfe", None) != 1:
        raise RuntimeError("MeanFlow internal nfe counter != 1.")
    print("OK: MeanFlow one-step inference verified.")


if __name__ == "__main__":
    main()
