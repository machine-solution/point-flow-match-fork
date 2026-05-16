#!/usr/bin/env python3
"""
Bar chart: per-episode length — baseline model vs phase-conditioned single model.

- Baseline: one bar (total steps).
- Phase-conditioned: stacked by phases {0,1,2} computed from *executed* robot_state gripper trace.
- ✓/✗ over bars: success flag from the recordings JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

from pfp.common.phase_utils import compute_phase_labels_np
from pfp import REPO_DIRS


def _success_flag(ep: dict) -> bool:
    v = ep.get("success")
    return bool(v) if v is not None else False


def _load_phase_params(phased_ckpt_name: str | None) -> tuple[float, int]:
    # Defaults (match conf/phase_conditioning/* and phase_utils)
    thr = 0.5
    contact_window = 2
    if phased_ckpt_name is None:
        return thr, contact_window
    cfg_path = REPO_DIRS.CKPT / phased_ckpt_name / "config.yaml"
    if not cfg_path.exists():
        return thr, contact_window
    cfg = OmegaConf.load(cfg_path)
    pcfg = getattr(cfg, "phase_conditioning", None)
    if pcfg is None:
        return thr, contact_window
    thr = float(getattr(pcfg, "gripper_close_threshold", thr))
    contact_window = int(getattr(pcfg, "contact_window", contact_window))
    return thr, contact_window


def _phase_counts_from_episode(ep: dict, *, thr: float, contact_window: int) -> tuple[int, int, int]:
    """Return (c0,c1,c2) counts for executed steps based on robot_states gripper."""
    rs = ep.get("robot_states") or []
    steps = int(ep.get("steps") or len(rs) or 0)
    if not rs or steps <= 0:
        return 0, 0, 0
    rs_arr = np.asarray(rs, dtype=np.float32)
    rs_arr = rs_arr[:steps]
    phase, _ = compute_phase_labels_np(
        rs_arr,
        thr=thr,
        contact_window=contact_window,
        num_phases=3,
    )
    c0 = int((phase == 0).sum())
    c1 = int((phase == 1).sum())
    c2 = int((phase == 2).sum())
    return c0, c1, c2


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", type=Path, required=True, help="recordings JSON for baseline model")
    p.add_argument("--phased", type=Path, required=True, help="recordings JSON for phase-conditioned model")
    p.add_argument(
        "--phased-ckpt-name",
        type=str,
        default=None,
        help="Optional ckpt/<name>/config.yaml to read phase params (thr/contact_window)",
    )
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("recordings/episode_lengths_baseline_vs_phased.png"),
        help="Output PNG path",
    )
    args = p.parse_args()

    with args.baseline.open() as f:
        d0 = json.load(f)
    with args.phased.open() as f:
        d1 = json.load(f)

    eps0 = d0["episodes"]
    eps1 = d1["episodes"]
    n = min(len(eps0), len(eps1))
    if len(eps0) != len(eps1):
        print(f"warning: episode counts differ ({len(eps0)} vs {len(eps1)}), plotting first {n}")

    thr, contact_window = _load_phase_params(args.phased_ckpt_name)
    print(f"[phase] thr={thr} contact_window={contact_window} (from ckpt={args.phased_ckpt_name})")

    base_lens = [int(eps0[i]["steps"]) for i in range(n)]
    c0: list[int] = []
    c1: list[int] = []
    c2: list[int] = []
    for i in range(n):
        a, b, c = _phase_counts_from_episode(eps1[i], thr=thr, contact_window=contact_window)
        # Fallback if robot_states missing: put everything into phase 0
        if a + b + c == 0 and eps1[i].get("steps") is not None:
            a = int(eps1[i]["steps"])
        c0.append(a)
        c1.append(b)
        c2.append(c)

    succ_base = [_success_flag(eps0[i]) for i in range(n)]
    succ_phased = [_success_flag(eps1[i]) for i in range(n)]

    x = np.arange(n)
    w = 0.36
    fig, ax = plt.subplots(figsize=(max(8.0, n * 0.9), 5.8), layout="constrained")

    c_base = "#4a6fa5"
    c_p0 = "#2d8f47"  # pre
    c_p1 = "#d7b13b"  # contact
    c_p2 = "#c45c26"  # post
    c_ok = "#156b2c"
    c_fail = "#a32020"

    ax.bar(x - w / 2, base_lens, width=w, color=c_base, label="baseline (всего шагов)")
    ax.bar(x + w / 2, c0, width=w, color=c_p0, label="phased: phase 0")
    ax.bar(x + w / 2, c1, width=w, bottom=c0, color=c_p1, label="phased: phase 1")
    ax.bar(x + w / 2, c2, width=w, bottom=np.array(c0) + np.array(c1), color=c_p2, label="phased: phase 2")

    heights_base = np.array(base_lens, dtype=float)
    heights_phased = np.array(c0, dtype=float) + np.array(c1, dtype=float) + np.array(c2, dtype=float)
    pad = max(3.0, 0.04 * max(float(np.max(heights_base)), float(np.max(heights_phased)), 1.0))
    for i in range(n):
        ax.text(
            x[i] - w / 2,
            heights_base[i] + pad * 0.15,
            "✓" if succ_base[i] else "✗",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color=c_ok if succ_base[i] else c_fail,
        )
        ax.text(
            x[i] + w / 2,
            heights_phased[i] + pad * 0.15,
            "✓" if succ_phased[i] else "✗",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color=c_ok if succ_phased[i] else c_fail,
        )

    ymax = float(np.max(np.concatenate([heights_base, heights_phased]))) + pad * 1.35
    ax.set_ylim(0, ymax)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(n)])
    ax.set_xlabel("эпизод (индекс, seed=base_seed+i)")
    ax.set_ylabel("число шагов")
    ax.set_title("Длина эпизода: baseline vs phase-conditioned (0/1/2)  — ✓/✗ успех")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.35)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()

