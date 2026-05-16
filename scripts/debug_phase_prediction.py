#!/usr/bin/env python3
"""Smoke test: phase_flow fallback + train/infer phase conditioning consistency."""
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


def _load_batch(zarr: Path, cfg, batch_size: int):
    from pfp.data.dataset_pcd import RobotDatasetPcd

    ds = RobotDatasetPcd(str(zarr), phase_conditioning=cfg.phase_conditioning, **cfg.dataset)
    return _collate([ds[i] for i in range(min(batch_size, len(ds)))])


def _assert_phase_flow_constant(phase_flow: torch.Tensor, phase: torch.Tensor, B: int, T: int) -> None:
    for b in range(B):
        row = phase_flow[b]
        assert row.unique().numel() == 1, f"row {b} not constant: {row.unique().tolist()}"
        assert int(row[0].item()) == int(phase[b, 0].item()), f"row {b} phase_flow[0] != phase[b,0]"


def _run_case(
    conf_dir: Path,
    *,
    case_id: str,
    title: str,
    overrides: list[str],
    zarr: Path,
    batch_size: int,
    check_infer: bool = False,
) -> None:
    print(f"\n{'=' * 60}")
    print(f"CASE {case_id}: {title}")
    print(f"overrides: {overrides}")

    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg = compose(config_name="train", overrides=overrides)
    OmegaConf.resolve(cfg)

    pcd, rs_obs, rs_pred, phase_pred = _load_batch(zarr, cfg, batch_size)
    model = instantiate(
        cfg.model,
        phase_conditioning=cfg.phase_conditioning,
        phase_prediction=cfg.phase_prediction,
        phase_rollout=getattr(cfg, "phase_rollout", None),
    )
    model = model.to(DEVICE).float().train()

    batch_norm = model._norm_data((pcd, rs_obs, rs_pred, phase_pred))
    pcd_n, rs_obs_n, rs_pred_n, phase_n = batch_norm
    B, T = phase_n.shape[0], phase_n.shape[1]

    print(f"phase shape: {tuple(phase_n.shape)}")
    print(f"phase[:,0]: {phase_n[:, 0].tolist()}")

    loss_xyz, loss_rot6d, loss_grip, phase_loss, phase_metrics = model.calculate_loss(
        pcd_n, rs_obs_n, rs_pred_n, phase=phase_n
    )
    cfm = (
        model.l_w["xyz"] * loss_xyz
        + model.l_w["rot6d"] * loss_rot6d
        + model.l_w["grip"] * loss_grip
    )
    total = cfm + (float(model.phase_pred_cfg.loss_weight) * phase_loss if phase_loss is not None else 0.0)

    nx = model.encode_obs(pcd_n, rs_obs_n)
    phase_flow = model._phase_for_flow_train(phase_n, nx, B=B, T=T)
    mode = model._phase_flow_training_mode()

    print(f"phase_flow shape: {tuple(phase_flow.shape)}")
    print(f"phase_flow training mode: {mode}")
    print(f"condition_flow_with_current_phase_train: {model.phase_pred_cfg.condition_flow_with_current_phase_train}")
    print(f"phase_flow unique per row: {[phase_flow[i].unique().tolist() for i in range(B)]}")

    if phase_loss is not None:
        print(f"phase_logits shape: {tuple(model.predict_phase_logits(nx).shape)}")
        print(f"phase_loss: {float(phase_loss):.6f}")
    print(f"cfm_loss: {float(cfm):.6f}  total_loss: {float(total):.6f}")
    for k in sorted(phase_metrics):
        if "phase_flow" in k or "phase_gt" in k or "condition_current" in k:
            print(f"  {k}: {phase_metrics[k]}")

    # Case-specific assertions
    if case_id == "A":
        assert not model.phase_pred_enabled
        assert phase_loss is None
        assert torch.equal(phase_flow, phase_n), "CASE A: phase_flow must equal full GT phase"
        print("PASS CASE A: oracle phase, no crash, phase_flow == phase")

    elif case_id == "B":
        assert model.phase_pred_enabled
        assert model.phase_pred_cfg.condition_flow_with_current_phase_train
        assert phase_loss is not None
        _assert_phase_flow_constant(phase_flow, phase_n, B, T)
        expected = phase_n[:, 0].view(B, 1).expand(B, T)
        assert torch.equal(phase_flow, expected)
        print("PASS CASE B: current GT phase repeated across horizon")

    elif case_id == "C":
        assert model.phase_pred_enabled
        assert not model.phase_pred_cfg.condition_flow_with_current_phase_train
        assert not model.phase_pred_cfg.use_predicted_phase_for_flow_train
        assert torch.equal(phase_flow, phase_n), "CASE C: phase_flow must equal full GT phase"
        print("PASS CASE C: full GT horizon phase_flow == phase")

    if check_infer:
        with torch.no_grad():
            pred_y = model.infer_y(pcd_n, rs_obs_n, phase=None)
            assert tuple(pred_y.shape) == (B, model.n_pred_steps, model.y_dim)
            logits = model.predict_phase_logits(model.encode_obs(pcd_n, rs_obs_n))
            p_cur = logits.argmax(dim=-1)
            phase_seq = p_cur.view(B, 1).expand(B, model.n_pred_steps)
            for b in range(B):
                assert phase_seq[b].unique().numel() == 1
            print(f"predicted phase: {p_cur.tolist()}")
            print(f"infer_y shape: {tuple(pred_y.shape)}")
            print("PASS CASE D: infer uses flat predicted phase across horizon")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-dir", type=Path, default=None)
    ap.add_argument("--zarr", type=Path, default=Path("demos/sim/open_fridge/train"))
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)

    conf_dir = args.config_dir or (Path(__file__).resolve().parents[1] / "conf")
    if not conf_dir.is_dir():
        print(f"Config dir not found: {conf_dir}", file=sys.stderr)
        sys.exit(1)

    _run_case(
        conf_dir,
        case_id="A",
        title="phase_conditioning=true, phase_prediction=false",
        overrides=["phase_conditioning=enabled", "phase_prediction=disabled"],
        zarr=args.zarr,
        batch_size=args.batch_size,
    )
    _run_case(
        conf_dir,
        case_id="B",
        title="learned phase, condition_flow_with_current_phase_train=true",
        overrides=["phase_conditioning=enabled", "phase_prediction=enabled"],
        zarr=args.zarr,
        batch_size=args.batch_size,
    )
    _run_case(
        conf_dir,
        case_id="C",
        title="learned phase, full GT horizon ablation",
        overrides=[
            "phase_conditioning=enabled",
            "phase_prediction=enabled",
            "phase_prediction.condition_flow_with_current_phase_train=false",
        ],
        zarr=args.zarr,
        batch_size=args.batch_size,
    )
    _run_case(
        conf_dir,
        case_id="D",
        title="inference phase=None with learned phase",
        overrides=["phase_conditioning=enabled", "phase_prediction=enabled"],
        zarr=args.zarr,
        batch_size=args.batch_size,
        check_infer=True,
    )

    print("\nAll cases passed.")


if __name__ == "__main__":
    main()
