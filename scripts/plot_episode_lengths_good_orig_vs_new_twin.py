#!/usr/bin/env python3
"""Bar chart: per-episode length — old model (single bar) vs two-phase (stacked pre/post)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _phase_counts(phases: list | None) -> tuple[int, int]:
    if not phases:
        return 0, 0
    pre = sum(1 for p in phases if p == "pre")
    post = sum(1 for p in phases if p == "post")
    return pre, post


def _success_flag(ep: dict) -> bool:
    v = ep.get("success")
    return bool(v) if v is not None else False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--good-orig",
        type=Path,
        default=Path("recordings/good_orig.json"),
        help="Aggregate JSON for single-model recordings",
    )
    p.add_argument(
        "--new-twin",
        type=Path,
        default=Path("recordings/new_twin.json"),
        help="Aggregate JSON for two-phase recordings",
    )
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("recordings/episode_lengths_good_orig_vs_new_twin.png"),
        help="Output PNG path",
    )
    args = p.parse_args()

    with args.good_orig.open() as f:
        d0 = json.load(f)
    with args.new_twin.open() as f:
        d1 = json.load(f)

    eps0 = d0["episodes"]
    eps1 = d1["episodes"]
    n = min(len(eps0), len(eps1))
    if len(eps0) != len(eps1):
        print(f"warning: episode counts differ ({len(eps0)} vs {len(eps1)}), plotting first {n}")

    old_lens = [int(eps0[i]["steps"]) for i in range(n)]
    pre_lens: list[int] = []
    post_lens: list[int] = []
    for i in range(n):
        pre, post = _phase_counts(eps1[i].get("phases"))
        if pre + post == 0 and eps1[i].get("steps") is not None:
            sw = eps1[i].get("switch_step")
            total = int(eps1[i]["steps"])
            if sw is not None:
                pre, post = int(sw), total - int(sw)
        pre_lens.append(pre)
        post_lens.append(post)

    succ_old = [_success_flag(eps0[i]) for i in range(n)]
    succ_new = [_success_flag(eps1[i]) for i in range(n)]

    x = np.arange(n)
    w = 0.36

    fig, ax = plt.subplots(figsize=(max(8.0, n * 0.9), 5.8), layout="constrained")

    c_old = "#4a6fa5"
    c_pre = "#2d8f47"
    c_post = "#c45c26"
    c_ok = "#156b2c"
    c_fail = "#a32020"

    ax.bar(x - w / 2, old_lens, width=w, color=c_old, label="старая модель (всего шагов)")
    ax.bar(x + w / 2, pre_lens, width=w, color=c_pre, label="новая: pre")
    ax.bar(x + w / 2, post_lens, width=w, bottom=pre_lens, color=c_post, label="новая: post")

    heights_old = np.array(old_lens, dtype=float)
    heights_new = np.array(pre_lens, dtype=float) + np.array(post_lens, dtype=float)
    pad = max(3.0, 0.04 * max(float(np.max(heights_old)), float(np.max(heights_new)), 1.0))
    for i in range(n):
        ax.text(
            x[i] - w / 2,
            heights_old[i] + pad * 0.15,
            "✓" if succ_old[i] else "✗",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color=c_ok if succ_old[i] else c_fail,
        )
        ax.text(
            x[i] + w / 2,
            heights_new[i] + pad * 0.15,
            "✓" if succ_new[i] else "✗",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color=c_ok if succ_new[i] else c_fail,
        )

    ymax = float(np.max(np.concatenate([heights_old, heights_new]))) + pad * 1.35
    ax.set_ylim(0, ymax)

    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(n)])
    ax.set_xlabel("эпизод (индекс)")
    ax.set_ylabel("число шагов")
    ax.set_title(
        "Длина эпизода: good_orig vs new_twin (pre + post). "
        "✓/✗ над столбцами: слева — старая модель, справа — новая (поле success)"
    )
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.35)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
