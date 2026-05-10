#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf


def _as_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", type=Path, default=Path("demos/sim/open_fridge/train"))
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--phase", type=str, default="enabled", choices=["enabled", "disabled"])
    args = ap.parse_args()

    # Ensure diffusion_policy is importable if running locally
    # (users typically set PYTHONPATH=../diffusion_policy)
    from pfp.data.dataset_pcd import RobotDatasetPcd
    from pfp.policy.fm_policy import FMPolicy

    train_cfg = OmegaConf.load("conf/train.yaml")
    phase_cfg = OmegaConf.load(f"conf/phase_conditioning/{args.phase}.yaml")

    ds = RobotDatasetPcd(str(args.zarr), phase_conditioning=phase_cfg, **train_cfg.dataset)
    batch = [ds[i] for i in range(args.batch_size)]

    # naive collate
    def stack(i):
        return torch.stack([_as_tensor(b[i]) for b in batch], dim=0)

    pcd = stack(0)
    rs_obs = stack(1)
    rs_pred = stack(2)
    phase = stack(3) if (phase_cfg.enabled and len(batch[0]) >= 4) else None

    print("[batch] pcd", tuple(pcd.shape), pcd.dtype)
    print("[batch] rs_obs", tuple(rs_obs.shape), rs_obs.dtype)
    print("[batch] rs_pred", tuple(rs_pred.shape), rs_pred.dtype)
    if phase is not None:
        print("[batch] phase", tuple(phase.shape), phase.dtype, "min/max", int(phase.min()), int(phase.max()))

    # Instantiate model components without relying on unresolved Hydra interpolations.
    from pfp.backbones.pointnet import PointNetBackbone
    from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

    backbone = PointNetBackbone(embed_dim=256, input_channels=3, input_transform=False, use_group_norm=False)
    input_dim = 10 + (int(phase_cfg.phase_embed_dim) if bool(phase_cfg.enabled) else 0)
    diffusion_net = ConditionalUnet1D(
        input_dim=input_dim,
        global_cond_dim=266 * int(train_cfg.n_obs_steps),
        diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024],
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
        use_dropout=False,
    )
    model = FMPolicy(
        x_dim=266,
        y_dim=10,
        n_obs_steps=int(train_cfg.n_obs_steps),
        n_pred_steps=int(train_cfg.n_pred_steps),
        num_k_infer=2,
        time_conditioning=True,
        obs_encoder=backbone,
        diffusion_net=diffusion_net,
        loss_weights={"xyz": 10.0, "rot6d": 10.0, "grip": 1.0},
        norm_pcd_center=[0.4, 0.0, 1.4],
        phase_conditioning=phase_cfg,
    )
    model = model.float()

    with torch.no_grad():
        # forward shapes
        if phase is not None:
            loss_xyz, loss_rot6d, loss_grip = model.calculate_loss(pcd, rs_obs, rs_pred, phase=phase)
        else:
            loss_xyz, loss_rot6d, loss_grip = model.calculate_loss(pcd, rs_obs, rs_pred, phase=None)
        print("[loss] xyz", float(loss_xyz), "rot6d", float(loss_rot6d), "grip", float(loss_grip))

        pred = model.infer_y(pcd, rs_obs, phase=phase)
        print("[infer_y] out", tuple(pred.shape))


if __name__ == "__main__":
    main()

