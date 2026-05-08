#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from pfp.common.phase_utils import compute_phase_labels_np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", type=Path, required=True, help="Path to zarr dataset (directory with data/meta)")
    ap.add_argument("--episodes", type=str, default="0,1,2", help="Comma-separated episode indices to inspect")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--contact-window", type=int, default=2)
    ap.add_argument("--out", type=Path, default=None, help="Optional path to save plot PNG")
    args = ap.parse_args()

    # Import here so this script can run even if diffusion_policy isn't on sys.path by default.
    from pfp.data.replay_buffer import RobotReplayBuffer

    rb = RobotReplayBuffer.create_from_path(str(args.zarr), mode="r")
    ep_ids = [int(x.strip()) for x in args.episodes.split(",") if x.strip()]

    fig, axes = plt.subplots(len(ep_ids), 1, figsize=(10, 2.8 * max(1, len(ep_ids))), sharex=False)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])

    for ax, ep_i in zip(axes, ep_ids):
        ep = rb.get_episode(ep_i, copy=False)
        rs = np.asarray(ep["robot_state"], dtype=np.float32)
        g = rs[:, 9]
        phase, t_grasp = compute_phase_labels_np(
            rs, thr=float(args.thr), contact_window=int(args.contact_window), num_phases=3
        )
        counts = [(phase == k).sum() for k in range(3)]
        print(f"ep {ep_i}: T={len(rs)} grasp_t={t_grasp} phase_counts={counts}")
        ax.plot(g, label="gripper_open (raw)")
        ax.set_title(f"ep {ep_i}  grasp_t={t_grasp}")
        # show phase as background bands
        for k, color in [(0, "#dbeafe"), (1, "#fde68a"), (2, "#dcfce7")]:
            mask = phase == k
            if not mask.any():
                continue
            idx = np.where(mask)[0]
            # contiguous segments
            start = idx[0]
            prev = idx[0]
            for cur in idx[1:]:
                if cur != prev + 1:
                    ax.axvspan(start, prev, color=color, alpha=0.35, linewidth=0)
                    start = cur
                prev = cur
            ax.axvspan(start, prev, color=color, alpha=0.35, linewidth=0)
        if t_grasp is not None:
            ax.axvline(int(t_grasp), color="k", linestyle="--", linewidth=1, alpha=0.8, label="grasp_t")
        ax.set_ylabel("gripper")
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("timestep")
    plt.tight_layout()
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.out, dpi=150)
        print(f"wrote {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

