#!/usr/bin/env python3
"""Sanity check: ConditionalUnet1D input channels vs phase conditioning; one forward pass."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf


def _expected_input_dim(*, y_dim: int, phase_enabled: bool, phase_embed_dim: int) -> int:
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
        help="Path to conf/ (default: ../conf relative to repo root)",
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
        exp_in = _expected_input_dim(
            y_dim=y_dim, phase_enabled=enabled, phase_embed_dim=embed_d
        )

        model = instantiate(cfg.model, phase_conditioning=pcfg)
        model = model.float()
        actual_in = int(getattr(model, "diffusion_net_input_dim", -1))

        out_ch = None
        fc = getattr(model.diffusion_net, "final_conv", None)
        if fc is not None:
            last = list(fc.children())[-1]
            if hasattr(last, "out_channels"):
                out_ch = int(last.out_channels)

        print(f"\n=== {label} ===")
        print(f"  y_dim={y_dim}  phase_enabled={enabled}  phase_embed_dim={embed_d}")
        print(f"  expected diffusion_net input_dim = {exp_in}")
        print(f"  actual   diffusion_net_input_dim   = {actual_in}")
        print(f"  UNet final_conv out_channels (if any) = {out_ch}")

        if actual_in != exp_in:
            print("  ERROR: input_dim mismatch", file=sys.stderr)
            sys.exit(1)
        if out_ch is not None and out_ch != y_dim:
            print(f"  ERROR: output channels {out_ch} != y_dim {y_dim}", file=sys.stderr)
            sys.exit(1)

        B = 2
        T = int(cfg.n_pred_steps)
        npt = int(cfg.dataset.n_points)
        pcd = torch.randn(B, npt, 3)
        rs_obs = torch.randn(B, int(cfg.n_obs_steps), y_dim)
        rs_pred = torch.randn(B, T, y_dim)
        phase = torch.randint(0, int(getattr(pcfg, "num_phases", 3)), (B, T), dtype=torch.long)

        with torch.no_grad():
            if enabled:
                loss_xyz, loss_rot6d, loss_grip = model.calculate_loss(
                    pcd, rs_obs, rs_pred, phase=phase
                )
                pred = model.infer_y(pcd, rs_obs, phase=phase)
            else:
                loss_xyz, loss_rot6d, loss_grip = model.calculate_loss(
                    pcd, rs_obs, rs_pred, phase=None
                )
                pred = model.infer_y(pcd, rs_obs, phase=None)

        print(f"  calculate_loss ok (xyz={float(loss_xyz):.4f} ...)")
        print(f"  infer_y output shape = {tuple(pred.shape)}  (expect ({B}, {T}, {y_dim}))")
        if tuple(pred.shape) != (B, T, y_dim):
            print("  ERROR: infer_y shape mismatch", file=sys.stderr)
            sys.exit(1)

    print("\nOK: baseline and phase models built; UNet I/O dims match expectations.")


if __name__ == "__main__":
    main()
