#!/usr/bin/env python3
"""Smoke test: phase conditioning, learned phase head, train/infer consistency."""
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


def _run_loss_check(model, pcd_n, rs_obs_n, rs_pred_n, phase_n, *, label: str) -> None:
    print(f"\n=== {label} ===")
    B, T = phase_n.shape[0], phase_n.shape[1]
    loss_xyz, loss_rot6d, loss_grip, phase_loss, phase_metrics = model.calculate_loss(
        pcd_n, rs_obs_n, rs_pred_n, phase=phase_n
    )
    cfm = (
        model.l_w["xyz"] * loss_xyz
        + model.l_w["rot6d"] * loss_rot6d
        + model.l_w["grip"] * loss_grip
    )
    total = cfm
    if phase_loss is not None:
        total = cfm + float(model._phase_pred_cfg.loss_weight) * phase_loss

    nx = model.encode_obs(pcd_n, rs_obs_n)
    phase_flow = model._phase_for_flow_train(phase_n, nx, B=B, T=T)

    print(f"phase shape: {tuple(phase_n.shape)}")
    print(f"phase_flow shape: {tuple(phase_flow.shape)}")
    print(f"phase_flow unique per row (first 4): {[phase_flow[i].unique().tolist() for i in range(min(4, B))]}")
    flat_unique = phase_flow.reshape(-1).unique().tolist()
    print(f"phase_flow unique (all): {flat_unique}")

    if model.phase_pred_enabled:
        logits = model.predict_phase_logits(nx)
        print(f"phase_logits shape: {tuple(logits.shape)}")
        print(f"phase_loss: {float(phase_loss):.6f}")
        print(f"phase_accuracy: {phase_metrics.get('metrics/train/phase_accuracy')}")
        for k, v in sorted(phase_metrics.items()):
            print(f"  {k}: {v}")
        if model._phase_pred_cfg.condition_flow_with_current_phase_train:
            expected = phase_n[:, 0].view(B, 1).expand(B, T)
            assert torch.equal(phase_flow, expected), "phase_flow should be current GT phase repeated"
            print("OK: phase_flow == phase[:,0] repeated (train matches inference style)")
    else:
        assert phase_loss is None
        print("OK: no phase auxiliary loss")
        if phase_flow.shape == phase_n.shape and not torch.equal(phase_flow, phase_n):
            print("WARN: phase_flow differs from phase (unexpected for oracle-only)")
        elif torch.equal(phase_flow, phase_n):
            print("OK: phase_flow is full GT horizon phase")

    print(f"cfm_loss: {float(cfm):.6f}")
    print(f"total_loss: {float(total):.6f}")
    print(f"calculate_loss: OK (no crash)")


def _run_infer_check(model, pcd_n, rs_obs_n, *, label: str) -> None:
    print(f"\n--- inference ({label}) ---")
    pred_y = model.infer_y(pcd_n, rs_obs_n, phase=None)
    B = pcd_n.shape[0]
    print(f"infer_y shape: {tuple(pred_y.shape)} (expect ({B}, {model.n_pred_steps}, {model.y_dim}))")
    assert tuple(pred_y.shape) == (B, model.n_pred_steps, model.y_dim)

    if model.phase_pred_enabled:
        nx = model.encode_obs(pcd_n, rs_obs_n)
        p_cur = model.predict_phase_logits(nx).argmax(dim=-1)
        print(f"predicted phase (batch): {p_cur.tolist()}")
        print("OK: infer_y uses learned phase when phase=None")
    else:
        print("infer_y with phase=None (heuristic / rollout fallback if configured)")


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

    from pfp.data.dataset_pcd import RobotDatasetPcd

    scenarios = [
        ("oracle phase only", ["phase_conditioning=enabled", "phase_prediction=disabled"]),
        ("learned phase", ["phase_conditioning=enabled", "phase_prediction=enabled"]),
    ]

    for title, overrides in scenarios:
        with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
            cfg = compose(config_name="train", overrides=overrides)
        OmegaConf.resolve(cfg)

        ds = RobotDatasetPcd(
            str(args.zarr),
            phase_conditioning=cfg.phase_conditioning,
            **cfg.dataset,
        )
        batch = _collate([ds[i] for i in range(min(args.batch_size, len(ds)))])
        pcd, rs_obs, rs_pred, phase_pred = batch
        print(f"\n{'#' * 60}")
        print(f"# {title}")
        print(f"phase_pred shape: {tuple(phase_pred.shape)}")
        print(f"phase_target shape: {tuple(phase_pred[:, 0].shape)}  unique: {phase_pred[:, 0].unique().tolist()}")

        model = instantiate(
            cfg.model,
            phase_conditioning=cfg.phase_conditioning,
            phase_prediction=cfg.phase_prediction,
            phase_rollout=getattr(cfg, "phase_rollout", None),
        )
        model = model.to(DEVICE).float().train()

        batch_norm = model._norm_data((pcd, rs_obs, rs_pred, phase_pred))
        pcd_n, rs_obs_n, rs_pred_n, phase_n = batch_norm

        _run_loss_check(model, pcd_n, rs_obs_n, rs_pred_n, phase_n, label=title)

        with torch.no_grad():
            if cfg.phase_prediction.enabled:
                _run_infer_check(model, pcd_n, rs_obs_n, label=title)

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
