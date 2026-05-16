#!/usr/bin/env python3
"""Sanity check: phase-conditioned flow — UNet I/O width, sliced velocity, D-only state."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from pfp.policy.fm_policy import _unet_final_out_channels, _unet_stem_in_channels


def _expected_io_dim(*, y_dim: int, phase_enabled: bool, phase_embed_dim: int) -> int:
    d = int(y_dim)
    if phase_enabled:
        d += int(phase_embed_dim)
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Path to conf/ (default: repo/conf)",
    )
    args = ap.parse_args()

    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)

    repo_root = Path(__file__).resolve().parents[1]
    conf_dir = args.config_dir or (repo_root / "conf")
    if not conf_dir.is_dir():
        print(f"Config dir not found: {conf_dir}", file=sys.stderr)
        sys.exit(1)

    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg_b = compose(config_name="train", overrides=["phase_conditioning=disabled"])
        cfg_p = compose(config_name="train", overrides=["phase_conditioning=enabled"])

    for label, cfg in [("baseline (phase off)", cfg_b), ("phase-conditioned", cfg_p)]:
        OmegaConf.resolve(cfg)
        pcfg = cfg.phase_conditioning
        enabled = bool(getattr(pcfg, "enabled", False))
        embed_d = int(getattr(pcfg, "phase_embed_dim", 0))
        y_dim = int(cfg.y_dim)
        exp_io = _expected_io_dim(
            y_dim=y_dim, phase_enabled=enabled, phase_embed_dim=embed_d
        )

        model = instantiate(
            cfg.model,
            phase_conditioning=pcfg,
            phase_prediction=getattr(cfg, "phase_prediction", None),
            phase_rollout=getattr(cfg, "phase_rollout", None),
        )
        model = model.float()
        assert isinstance(model.diffusion_net, nn.Module), "diffusion_net must be nn.Module"
        ctor = getattr(model, "_diffusion_net_ctor_source", "?")
        print(f"  ctor: {ctor}")
        actual_in_attr = int(getattr(model, "diffusion_net_input_dim", -1))
        stem_in = _unet_stem_in_channels(model.diffusion_net)
        final_out = _unet_final_out_channels(model.diffusion_net)
        print(f"  UNet stem in_channels = {stem_in}")
        print(f"  UNet final_conv out_channels = {final_out}")

        print(f"\n=== {label} ===")
        print(
            f"  y_dim (D)={y_dim}  phase_enabled={enabled}  "
            f"phase_embed_dim (used if enabled)={(embed_d if enabled else 0)}"
        )
        print(f"  expected full field width (D or D+E) = {exp_io}")
        print(f"  model.diffusion_net_input_dim      = {actual_in_attr}")

        if actual_in_attr != exp_io:
            print("  ERROR: diffusion_net_input_dim mismatch", file=sys.stderr)
            sys.exit(1)
        if stem_in is not None and stem_in != exp_io:
            print("  ERROR: stem in_channels mismatch", file=sys.stderr)
            sys.exit(1)
        if final_out is not None and final_out != exp_io:
            print(
                f"  ERROR: final out_channels {final_out} != expected full width {exp_io}",
                file=sys.stderr,
            )
            sys.exit(1)

        B = 2
        T = int(cfg.n_pred_steps)
        n_obs = int(cfg.n_obs_steps)
        npt = int(cfg.dataset.n_points)
        pcd = torch.randn(B, n_obs, npt, 3)
        rs_obs = torch.randn(B, n_obs, y_dim)
        rs_pred = torch.randn(B, T, y_dim)
        phase = torch.randint(0, int(getattr(pcfg, "num_phases", 3)), (B, T), dtype=torch.long)

        with torch.no_grad():
            nx = model.obs_encoder(pcd, rs_obs)
            t = torch.rand(B, 1, 1, device=nx.device)
            z0 = torch.randn(B, T, y_dim, device=nx.device)
            z1 = rs_pred.to(nx.device)
            z_flow = t * z1 + (1.0 - t) * z0
            assert z_flow.shape[-1] == y_dim
            if enabled:
                pe = model.phase_embedding(phase.to(nx.device))
                z_cond = torch.cat([z_flow, pe], dim=-1)
                assert z_cond.shape[-1] == y_dim + embed_d
            else:
                z_cond = z_flow
                assert z_cond.shape[-1] == y_dim
            ts = t.squeeze(-1).squeeze(-1) * model.pos_emb_scale if model.time_conditioning else None
            pred_full = model.diffusion_net(z_cond, ts, global_cond=nx)
            pred_vel = pred_full[..., :y_dim]
            print(f"  [probe] z_flow {tuple(z_flow.shape)}  z_cond {tuple(z_cond.shape)}")
            print(f"  [probe] pred_full {tuple(pred_full.shape)}  pred_vel {tuple(pred_vel.shape)}")
            assert pred_full.shape[-1] == exp_io
            assert pred_vel.shape[-1] == y_dim

            if enabled:
                loss_xyz, loss_rot6d, loss_grip, _, _ = model.calculate_loss(
                    pcd, rs_obs, rs_pred, phase=phase
                )
                pred = model.infer_y(pcd, rs_obs, phase=phase)
            else:
                loss_xyz, loss_rot6d, loss_grip, _, _ = model.calculate_loss(
                    pcd, rs_obs, rs_pred, phase=None
                )
                pred = model.infer_y(pcd, rs_obs, phase=None)

        print(f"  calculate_loss ok (xyz={float(loss_xyz):.4f} ...)")
        print(f"  infer_y state shape = {tuple(pred.shape)}  (expect ({B}, {T}, {y_dim}), D-only)")
        if tuple(pred.shape) != (B, T, y_dim):
            print("  ERROR: infer_y shape mismatch", file=sys.stderr)
            sys.exit(1)

    print("\nOK: baseline and phase models; UNet full width matches input; velocity/state stay D.")


if __name__ == "__main__":
    main()
