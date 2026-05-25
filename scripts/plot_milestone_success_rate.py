#!/usr/bin/env python3
"""Plot success rate vs training epoch with binomial confidence intervals."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta


def clopper_pearson(k: int, n: int, alpha: float) -> tuple[float, float]:
    """Exact binomial CI for proportion k/n (Clopper–Pearson)."""
    if n <= 0:
        return 0.0, 0.0
    k = int(k)
    n = int(n)
    if k == 0:
        lo = 0.0
    else:
        lo = float(beta.ppf(alpha / 2, k, n - k + 1))
    if k == n:
        hi = 1.0
    else:
        hi = float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def load_runs(summary_path: Path) -> list[dict]:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    runs = []
    for r in data["runs"]:
        runs.append(
            {
                "epoch": int(r["train_epoch"]),
                "k": int(r["num_success"]),
                "n": int(r["num_episodes"]),
                "p": float(r["accuracy"]),
            }
        )
    runs.sort(key=lambda x: x["epoch"])
    return runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--summaries",
        nargs="+",
        required=True,
        help="Paths to summary.json (label inferred from ckpt_name).",
    )
    ap.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional display labels, same order as --summaries.",
    )
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--title", default="Success rate vs training epoch (open_fridge)")
    ap.add_argument("--seed", type=int, default=None, help="Shown in subtitle if set.")
    args = ap.parse_args()

    if args.labels and len(args.labels) != len(args.summaries):
        raise SystemExit("--labels count must match --summaries")

    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea"]
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for i, spath in enumerate(args.summaries):
        spath = Path(spath).resolve()
        meta = json.loads(spath.read_text(encoding="utf-8"))
        label = (
            args.labels[i]
            if args.labels
            else ("baseline" if "baseline" in meta["ckpt_name"] else "phased")
        )
        runs = load_runs(spath)
        epochs = np.array([r["epoch"] for r in runs])
        p = np.array([r["p"] for r in runs])
        lo99 = np.array([clopper_pearson(r["k"], r["n"], 0.01)[0] for r in runs])
        hi99 = np.array([clopper_pearson(r["k"], r["n"], 0.01)[1] for r in runs])
        lo95 = np.array([clopper_pearson(r["k"], r["n"], 0.05)[0] for r in runs])
        hi95 = np.array([clopper_pearson(r["k"], r["n"], 0.05)[1] for r in runs])

        c = colors[i % len(colors)]
        ax.fill_between(
            epochs,
            lo99,
            hi99,
            color=c,
            alpha=0.12,
            label=f"{label} 99% CI",
        )
        ax.fill_between(
            epochs,
            lo95,
            hi95,
            color=c,
            alpha=0.22,
            label=f"{label} 95% CI",
        )
        ax.plot(
            epochs,
            p,
            "o-",
            color=c,
            linewidth=2,
            markersize=8,
            label=f"{label} (n={runs[0]['n']}/epoch)",
        )
        for ep, rate, k, n in zip(epochs, p, [r["k"] for r in runs], [r["n"] for r in runs]):
            ax.annotate(
                f"{100 * rate:.0f}% ({k}/{n})",
                (ep, rate),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
                color=c,
            )

    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Success rate")
    ax.set_ylim(0, max(0.55, ax.get_ylim()[1]))
    ax.set_xticks([300, 600, 900, 1200, 1500])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    subtitle_parts = []
    if args.seed is not None:
        subtitle_parts.append(f"seed={args.seed}")
    meta0 = json.loads(Path(args.summaries[0]).read_text(encoding="utf-8"))
    if meta0.get("max_episode_length"):
        subtitle_parts.append(f"max_steps={meta0['max_episode_length']}")
    if subtitle_parts:
        ax.set_title(args.title + "\n" + ", ".join(subtitle_parts), fontsize=11)
    else:
        ax.set_title(args.title, fontsize=11)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
