from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class PhaseConditioningConfig:
    enabled: bool = False
    num_phases: int = 3
    phase_embed_dim: int = 32
    gripper_close_threshold: float = 0.5
    contact_window: int = 2


def phase_cfg_from(cfg: Any | None) -> PhaseConditioningConfig:
    """Best-effort conversion from OmegaConf/dict/None to a typed config."""
    if cfg is None:
        return PhaseConditioningConfig()
    # OmegaConf behaves like dict for get()
    enabled = bool(getattr(cfg, "enabled", None) if hasattr(cfg, "enabled") else cfg.get("enabled", False))
    num_phases = int(getattr(cfg, "num_phases", None) if hasattr(cfg, "num_phases") else cfg.get("num_phases", 3))
    phase_embed_dim = int(
        getattr(cfg, "phase_embed_dim", None) if hasattr(cfg, "phase_embed_dim") else cfg.get("phase_embed_dim", 32)
    )
    thr = float(
        getattr(cfg, "gripper_close_threshold", None)
        if hasattr(cfg, "gripper_close_threshold")
        else cfg.get("gripper_close_threshold", 0.5)
    )
    w = int(getattr(cfg, "contact_window", None) if hasattr(cfg, "contact_window") else cfg.get("contact_window", 2))
    return PhaseConditioningConfig(
        enabled=enabled,
        num_phases=num_phases,
        phase_embed_dim=phase_embed_dim,
        gripper_close_threshold=thr,
        contact_window=w,
    )


def find_first_grasp_index_from_robot_state(robot_state_seq: np.ndarray, *, thr: float) -> int | None:
    """Return first timestep where gripper channel (idx 9) is < thr, else None."""
    rs = np.asarray(robot_state_seq)
    if rs.ndim != 2 or rs.shape[1] < 10:
        raise ValueError(f"robot_state_seq must be (T,>=10), got {rs.shape}")
    g = rs[:, 9].astype(np.float64)
    closed = g < float(thr)
    if not np.any(closed):
        return None
    return int(np.argmax(closed))


def compute_phase_labels_np(
    robot_state_seq: np.ndarray,
    *,
    thr: float = 0.5,
    contact_window: int = 2,
    num_phases: int = 3,
) -> tuple[np.ndarray, int | None]:
    """
    Compute per-timestep phase labels for a full sequence of robot_state.

    Phases:
      0 = approach/pre-grasp
      1 = contact window around first close
      2 = manipulation/post-grasp
    """
    T = int(np.asarray(robot_state_seq).shape[0])
    if T <= 0:
        return np.zeros((0,), dtype=np.int64), None
    num_phases = int(num_phases)
    if num_phases != 3:
        raise ValueError(f"Only num_phases=3 supported by this heuristic, got {num_phases}")
    w = max(0, int(contact_window))
    t_grasp = find_first_grasp_index_from_robot_state(robot_state_seq, thr=thr)
    phase = np.zeros((T,), dtype=np.int64)
    if t_grasp is None:
        # no grasp detected -> keep phase 0 everywhere (approach)
        return phase, None
    lo = max(0, int(t_grasp) - w)
    hi = min(T, int(t_grasp) + w + 1)
    phase[lo:hi] = 1
    phase[hi:] = 2
    return phase, int(t_grasp)


def compute_phase_labels_torch_from_gripper(
    gripper_seq: torch.Tensor,
    *,
    thr: float,
    contact_window: int,
    num_phases: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized phase labels for a batch of gripper sequences.

    gripper_seq: (B, T) float tensor
    returns:
      phase: (B, T) int64 in [0,2]
      grasp_idx: (B,) int64, -1 if no grasp detected
    """
    if num_phases != 3:
        raise ValueError(f"Only num_phases=3 supported by this heuristic, got {num_phases}")
    g = gripper_seq
    if g.ndim != 2:
        raise ValueError(f"gripper_seq must be (B,T), got {tuple(g.shape)}")
    B, T = g.shape
    closed = g < float(thr)
    any_closed = closed.any(dim=1)
    # first index of True; if none -> 0, then mask to -1
    first = torch.argmax(closed.to(torch.int64), dim=1)
    grasp_idx = torch.where(any_closed, first.to(torch.int64), torch.full((B,), -1, device=g.device, dtype=torch.int64))
    phase = torch.zeros((B, T), device=g.device, dtype=torch.int64)
    w = int(max(0, contact_window))
    if T == 0:
        return phase, grasp_idx
    # build window mask around grasp_idx
    tt = torch.arange(T, device=g.device).view(1, T)
    gi = grasp_idx.view(B, 1)
    in_window = (gi >= 0) & (tt >= (gi - w)) & (tt <= (gi + w))
    after_window = (gi >= 0) & (tt > (gi + w))
    phase = torch.where(in_window, torch.ones_like(phase), phase)
    phase = torch.where(after_window, torch.full_like(phase, 2), phase)
    return phase, grasp_idx

