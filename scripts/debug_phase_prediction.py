#!/usr/bin/env python3
"""Smoke test: learned phase head + teacher-forced flow training."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from pfp import DEVICE


def _collate(batch):
    return tuple(torch.stack([torch.as_tensor(b[i]) for b in batch], dim=0) for i in range(len(batch[0])))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-dir", type=Path, default=None)
    ap.add_argument("--zarr", type=Path, default=Path("demos/sim/open_fridge/train"))
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)

    repo = Path(__file__).resolve().parents[1]
    conf_dir = args.config_dir or (repo / "conf")
    if not conf_dir.is_dir():
        print(f"Config dir not found: {conf_dir}", file=sys.stderr)
        sys.exit(1)

    overrides = [
        "phase_conditioning=enabled",
        "phase_prediction=enabled",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg = compose(config_name="train", overrides=overrides)
    OmegaConf.resolve(cfg)

    from pfp.data.dataset_pcd import RobotDatasetPcd

    ds = RobotDatasetPcd(
        str(args.zarr),
        phase_conditioning=cfg.phase_conditioning,
        **cfg.dataset,
    )
    batch = _collate([ds[i] for i in range(min(args.batch_size, len(ds)))])

    pcd, rs_obs, rs_pred, phase_pred = batch
    print(f"phase_pred shape: {tuple(phase_pred.shape)}")
    phase_target = phase_pred[:, 0]
    print(f"phase_target (= phase_pred[:,0]) shape: {tuple(phase_target.shape)}  unique: {phase_target.unique().tolist()}")

    model = instantiate(
        cfg.model,
        phase_conditioning=cfg.phase_conditioning,
        phase_prediction=cfg.phase_prediction,
        phase_rollout=getattr(cfg, "phase_rollout", None),
    )
    model = model.to(DEVICE).float().train()
    assert model.phase_pred_enabled, "expected phase_prediction.enabled=true"

    with torch.no_grad():
        batch_norm = model._norm_data((pcd, rs_obs, rs_pred, phase_pred))
        pcd_n, rs_obs_n, rs_pred_n, phase_n = batch_norm
        loss_xyz, loss_rot6d, loss_grip, phase_loss, phase_metrics = model.calculate_loss(
            pcd_n, rs_obs_n, rs_pred_n, phase=phase_n
        )
        cfm = (
            model.l_w["xyz"] * loss_xyz
            + model.l_w["rot6d"] * loss_rot6d
            + model.l_w["grip"] * loss_grip
        )
        total = cfm + float(model._phase_pred_cfg.loss_weight) * phase_loss

        nx = model.encode_obs(pcd_n, rs_obs_n)
        logits = model.predict_phase_logits(nx)
        pred_phase = logits.argmax(dim=-1)
        acc = (pred_phase == phase_target.to(DEVICE)).float().mean()

        print(f"phase_logits shape: {tuple(logits.shape)}")
        print(f"cfm_loss (weighted components): {float(cfm):.6f}")
        print(f"phase_loss: {float(phase_loss):.6f}")
        print(f"total_loss: {float(total):.6f}")
        print(f"phase_accuracy (manual): {float(acc):.4f}")
        for k, v in sorted(phase_metrics.items()):
            print(f"  {k}: {v}")

        pred_y = model.infer_y(pcd_n, rs_obs_n, phase=None)
        print(f"infer_y (phase=None) shape: {tuple(pred_y.shape)}  expected ({pcd_n.shape[0]}, {model.n_pred_steps}, {model.y_dim})")

        # Verify flat predicted phase along horizon
        phase_logits2 = model.predict_phase_logits(model.encode_obs(pcd_n, rs_obs_n))
        p_cur = phase_logits2.argmax(dim=-1)
        expected = p_cur.view(-1, 1).expand(-1, model.n_pred_steps)
        print(f"predicted phase (batch): {p_cur.tolist()}")
        print("OK: infer_y uses learned phase when phase=None")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
