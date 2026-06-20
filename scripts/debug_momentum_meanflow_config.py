#!/usr/bin/env python3
"""Sanity checks for MomentumMeanFlow config and one train/infer step."""
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
        default=["+experiment=pointflowmatch_momentum_meanflow"],
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
    print(f"state_input_multiplier={getattr(model, 'state_input_multiplier', None)}")
    print(f"diffusion_net_input_dim={getattr(model, 'diffusion_net_input_dim', None)}")
    print(f"schedule={model.mm_cfg.momentum_meanflow_schedule}")
    print(f"lambda_correct={model.mm_cfg.lambda_correct}")
    print(f"start_case_ratio={model.mm_cfg.start_case_ratio}")

    b = 2
    t = int(cfg.n_pred_steps)
    d = int(cfg.y_dim)
    pcd = torch.randn(b, cfg.n_obs_steps, cfg.dataset.n_points, 3)
    robot_state_obs = torch.randn(b, cfg.n_obs_steps, d)
    robot_state_pred = torch.randn(b, t, d)
    print("===== Dummy Shapes =====")
    print(
        f"pcd={tuple(pcd.shape)} robot_state_obs={tuple(robot_state_obs.shape)} "
        f"robot_state_pred={tuple(robot_state_pred.shape)}"
    )

    class _NoopLogger:
        @staticmethod
        def log_metrics(_m):
            return None

    model.logger = _NoopLogger()
    model.train()
    loss = model.loss(None, (pcd, robot_state_obs, robot_state_pred))
    assert torch.isfinite(loss).all(), "loss is not finite"
    print(f"===== Train loss (finite) ===== loss={float(loss):.6f}")
    loss.backward()
    grad_norm = sum(
        float(p.grad.norm().item()) for p in model.parameters() if p.grad is not None
    )
    assert grad_norm > 0.0, "expected non-zero gradients after backward"
    print(f"===== Backward ok ===== grad_norm_sum={grad_norm:.6f}")
    model.zero_grad(set_to_none=True)

    # Shape checks (spec §9).
    b, t, dim = robot_state_pred.shape
    prev_u = torch.zeros_like(robot_state_pred)
    x_in = torch.cat([robot_state_pred, prev_u], dim=-1)
    assert robot_state_pred.shape == (b, t, dim)
    assert prev_u.shape == (b, t, dim)
    assert x_in.shape == (b, t, 2 * dim)
    assert int(model.diffusion_net_input_dim) == 2 * dim
    assert int(model.diffusion_net_output_dim) == dim
    print("===== Shape checks ok =====")

    # Correction target behavior (spec §9.5-9.6).
    with torch.no_grad():
        nx = model.encode_obs(pcd, robot_state_obs)
        z0 = model._init_noise(b)
        z1 = robot_state_pred
        t0 = torch.full((b, 1, 1), 0.1)
        t1 = torch.full((b, 1, 1), 0.4)
        t2 = torch.full((b, 1, 1), 0.9)
        dt01 = t1 - t0
        dt12 = t2 - t1
        x_t0 = (1.0 - t0) * z0 + t0 * z1
        x_t1 = (1.0 - t1) * z0 + t1 * z1
        x_t2 = (1.0 - t2) * z0 + t2 * z1
        u01_target = (x_t1 - x_t0) / dt01
        u01_pred = u01_target.clone()
        x_t1_hat = x_t0 + dt01 * u01_pred
        u12_target = (x_t2 - x_t1_hat) / dt12
        oracle_u12 = (x_t2 - x_t1) / dt12
        err_perfect = float((u12_target - oracle_u12).norm(dim=-1).mean().item())
        u01_bad = u01_target + 0.5
        x_t1_hat_bad = x_t0 + dt01 * u01_bad
        u12_bad_target = (x_t2 - x_t1_hat_bad) / dt12
        err_bad_vs_oracle = float((u12_bad_target - oracle_u12).norm(dim=-1).mean().item())
        err_bad_vs_z1z0 = float((u12_bad_target - (z1 - z0)).norm(dim=-1).mean().item())
    assert err_perfect < 1e-4, f"perfect u01 should match oracle u12 target, err={err_perfect}"
    assert err_bad_vs_oracle > 1e-3, f"bad u01 should change corrective target, err={err_bad_vs_oracle}"
    assert err_bad_vs_z1z0 > 1e-3, f"bad u01 target should differ from z1-z0, err={err_bad_vs_z1z0}"
    print(
        "===== Correction target checks ok ===== "
        f"perfect_err={err_perfect:.2e} bad_vs_oracle={err_bad_vs_oracle:.4f} bad_vs_z1z0={err_bad_vs_z1z0:.4f}"
    )

    model.eval()
    with torch.no_grad():
        out = model.infer_y(pcd, robot_state_obs)
    assert out.shape == (b, t, d), f"unexpected infer shape {tuple(out.shape)}"
    print(f"===== Infer K={model.num_k_infer} ===== last_infer_nfe={model.last_infer_nfe}")

    for profile_on in (False, True):
        model.enable_profile_inference(profile_on)
        for k in (1, 2, 5, 10):
            model.set_num_k_infer(k)
            with torch.no_grad():
                _ = model.infer_y(pcd, robot_state_obs)
            assert int(model.last_infer_nfe) == k, (
                f"profile={profile_on}: expected nfe={k}, got {model.last_infer_nfe}"
            )
            print(f"infer profile={profile_on} K={k}: nfe={model.last_infer_nfe} ok")
    model.enable_profile_inference(False)

    for sched in ("uniform", "fm_exp"):
        model.set_momentum_meanflow_schedule(sched)
        model.set_num_k_infer(10)
        with torch.no_grad():
            _ = model.infer_y(pcd, robot_state_obs)
        print(f"schedule={sched}: nfe={model.last_infer_nfe} ok")

    print("===== Sanity checks passed =====")


if __name__ == "__main__":
    main()
