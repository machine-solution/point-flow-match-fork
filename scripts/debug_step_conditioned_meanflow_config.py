#!/usr/bin/env python3
"""Debug Hydra composition/instantiation for StepConditionedMeanFlow config."""
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
        default=["+experiment=pointflowmatch_step_conditioned_meanflow"],
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
    print(
        "step_cfg="
        f"{getattr(getattr(model, 'scmf_cfg', None), 'train_delta_steps', None)} "
        f"lambda_consistency={getattr(getattr(model, 'scmf_cfg', None), 'lambda_consistency', None)}"
    )

    b = 2
    pcd = torch.randn(b, cfg.n_obs_steps, cfg.dataset.n_points, 3)
    robot_state_obs = torch.randn(b, cfg.n_obs_steps, cfg.y_dim)
    robot_state_pred = torch.randn(b, cfg.n_pred_steps, cfg.y_dim)
    print("===== Dummy Shapes =====")
    print(f"pcd={tuple(pcd.shape)} robot_state_obs={tuple(robot_state_obs.shape)} robot_state_pred={tuple(robot_state_pred.shape)}")

    class _NoopLogger:
        @staticmethod
        def log_metrics(_m):
            return None

    model.logger = _NoopLogger()
    with torch.no_grad():
        total = model.loss(None, (pcd, robot_state_obs, robot_state_pred))
    print("===== Dummy Train Loss =====")
    print(f"loss_total={float(total):.6f}")

    with torch.no_grad():
        _ = model.infer_y(pcd, robot_state_obs)
    print("===== Dummy Infer K=1 =====")
    print(f"last_infer_nfe={getattr(model, 'last_infer_nfe', None)}")

    model.set_num_k_infer(4)
    with torch.no_grad():
        _ = model.infer_y(pcd, robot_state_obs)
    print("===== Dummy Infer K=4 =====")
    print(f"last_infer_nfe={getattr(model, 'last_infer_nfe', None)}")


if __name__ == "__main__":
    main()
