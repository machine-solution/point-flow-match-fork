#!/usr/bin/env python3
"""Plot baseline vs MeanFlow success rate vs inference steps K with binomial CI."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from binomial_ci import clopper_pearson


def load_sweep_rows(csv_path: Path, method: str) -> list[dict]:
    rows: list[dict] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["method"] != method:
                continue
            rows.append(
                {
                    "k": int(row["k"]),
                    "success": int(row["success"]),
                    "n": int(row["num_episodes"]),
                    "rate": float(row["success_rate"]),
                }
            )
    rows.sort(key=lambda r: r["k"])
    if not rows:
        raise SystemExit(f"No rows with method={method!r} in {csv_path}")
    return rows


def _plot_series(
    ax,
    rows: list[dict],
    *,
    color: str,
    label: str,
    confidences: list[float],
    fill_alpha: dict[float, float],
    annotate: bool,
    annotate_dy: float,
) -> float:
    ks = np.array([r["k"] for r in rows])
    rates = np.array([r["rate"] for r in rows])
    hi_max = 0.0

    for conf in confidences:
        alpha = 1.0 - conf
        lo = np.array([clopper_pearson(r["success"], r["n"], alpha)[0] for r in rows])
        hi = np.array([clopper_pearson(r["success"], r["n"], alpha)[1] for r in rows])
        hi_max = max(hi_max, float(hi.max()))
        label_pct = f"{100 * conf:g}%" if conf != int(conf * 1000) / 1000 else f"{100 * conf:.1f}%"
        ax.fill_between(
            ks,
            lo,
            hi,
            color=color,
            alpha=fill_alpha.get(conf, 0.18),
            label=f"{label} {label_pct} CI",
        )

    ax.plot(
        ks,
        rates,
        "o-",
        color=color,
        linewidth=2,
        markersize=8,
        label=label,
    )

    if annotate:
        for k, rate, success, n in zip(ks, rates, [r["success"] for r in rows], [r["n"] for r in rows]):
            ax.annotate(
                f"{100 * rate:.0f}% ({success}/{n})",
                (k, rate),
                textcoords="offset points",
                xytext=(0, annotate_dy),
                ha="center",
                fontsize=7,
                color=color,
            )

    return hi_max


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        type=Path,
        default=repo_root / "results/efficiency/fm_vs_meanflow_k_sweep.csv",
    )
    ap.add_argument(
        "--confidence",
        type=float,
        nargs="+",
        default=[0.95],
        help="CI levels per series (default: 95%%). Use 0.995 0.95 for wider bands.",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=repo_root / "results/figures/fm_vs_meanflow_sr_vs_k.png",
    )
    ap.add_argument(
        "--title",
        default="Baseline vs MeanFlow: success rate vs inference steps",
    )
    args = ap.parse_args()

    confidences = sorted(set(args.confidence), reverse=True)
    fill_alpha = {0.995: 0.10, 0.99: 0.12, 0.95: 0.18}

    baseline = load_sweep_rows(args.csv.resolve(), "baseline")
    meanflow = load_sweep_rows(args.csv.resolve(), "meanflow_multistep")

    fig, ax = plt.subplots(figsize=(9, 5.5))

    hi_max = _plot_series(
        ax,
        baseline,
        color="#2563eb",
        label="Baseline FM",
        confidences=confidences,
        fill_alpha=fill_alpha,
        annotate=True,
        annotate_dy=12,
    )
    hi_max = max(
        hi_max,
        _plot_series(
            ax,
            meanflow,
            color="#dc2626",
            label="MeanFlow (multistep, uniform)",
            confidences=confidences,
            fill_alpha=fill_alpha,
            annotate=True,
            annotate_dy=-14,
        ),
    )

    ks = np.array([r["k"] for r in baseline])
    ax.set_xlabel("Inference steps K")
    ax.set_ylabel("Success rate")
    ax.set_xticks(ks)
    ax.set_ylim(0, min(1.0, hi_max + 0.10))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    n_ep = baseline[0]["n"]
    ax.set_title(
        f"{args.title}\n"
        f"open_fridge, n={n_ep} episodes/point, seed=5678, max_steps=120",
        fontsize=11,
    )

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    svg_path = args.output.with_suffix(".svg")
    fig.savefig(svg_path, bbox_inches="tight")
    print(f"Wrote {args.output}")
    print(f"Wrote {svg_path}")


if __name__ == "__main__":
    main()
