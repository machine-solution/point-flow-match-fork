#!/usr/bin/env python3
"""MeanFlow inference schedule plots: SR sweep and step sizes at K=10."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from pfp.common.meanflow_utils import (  # noqa: E402
    MEANFLOW_INFER_SCHEDULES,
    build_meanflow_time_grid,
    meanflow_grid_deltas,
)
from binomial_ci import clopper_pearson  # noqa: E402


def load_schedule_sweep(csv_path: Path) -> list[dict]:
    rows: list[dict] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "schedule": row["schedule"],
                    "k": int(row["k"]),
                    "success": int(row["success"]),
                    "n": int(row["num_episodes"]),
                    "rate": float(row["success_rate"]),
                }
            )
    rows.sort(key=lambda r: r["rate"], reverse=True)
    return rows


def plot_sr_by_schedule(
    rows: list[dict],
    *,
    output: Path,
    title: str,
    confidence: float,
) -> None:
    alpha = 1.0 - confidence
    schedules = [r["schedule"] for r in rows]
    rates = np.array([r["rate"] for r in rows])
    lo = np.array([clopper_pearson(r["success"], r["n"], alpha)[0] for r in rows])
    hi = np.array([clopper_pearson(r["success"], r["n"], alpha)[1] for r in rows])
    yerr = np.vstack([rates - lo, hi - rates])

    x = np.arange(len(schedules))
    fig, ax = plt.subplots(figsize=(9, 5))
    color = "#2563eb"
    bars = ax.bar(x, rates, color=color, alpha=0.85, width=0.65, label="Success rate")
    ax.errorbar(
        x,
        rates,
        yerr=yerr,
        fmt="none",
        ecolor="#1e3a5f",
        capsize=5,
        capthick=1.5,
        elinewidth=1.5,
        label=f"{100 * confidence:g}% CI (Clopper–Pearson)",
    )

    for i, r in enumerate(rows):
        ax.text(
            x[i],
            hi[i] + 0.012,
            f"{100 * r['rate']:.0f}% ({r['success']}/{r['n']})",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#1e3a5f",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(schedules, rotation=25, ha="right")
    ax.set_ylabel("Success rate")
    ax.set_xlabel("MeanFlow inference schedule")
    ax.set_ylim(0, min(1.0, float(hi.max()) + 0.10))
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    k = rows[0]["k"]
    n = rows[0]["n"]
    ax.set_title(
        f"{title}\n"
        f"MeanFlow multistep, K={k}, n={n} episodes/schedule, seed=5678",
        fontsize=11,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")


def plot_dt_panels_by_sr(
    rows: list[dict],
    *,
    k: int,
    exp_scale: float,
    output: Path,
    title: str,
    ncols: int = 2,
    extra_success: int = 0,
) -> None:
    """Panels (SR descending): SR label left, Δt bars right, shared color per schedule."""
    extra_success = max(0, int(extra_success))
    n = len(rows)
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n / ncols))
    colors = plt.colormaps["tab10"].resampled(n)
    step_idx = np.arange(1, k + 1)

    fig = plt.figure(figsize=(7.2 * ncols, 3.4 * nrows))
    outer = fig.add_gridspec(nrows, ncols, hspace=0.42, wspace=0.32)

    dt_max = 0.0
    for row in rows:
        grid = build_meanflow_time_grid(k, row["schedule"], exp_scale=exp_scale)
        dt_max = max(dt_max, max(meanflow_grid_deltas(grid)))

    for i, row in enumerate(rows):
        grid_row, grid_col = divmod(i, ncols)
        inner = outer[grid_row, grid_col].subgridspec(
            1, 2, width_ratios=[0.34, 0.66], wspace=0.22
        )
        color = colors(i)
        schedule = row["schedule"]
        n_ep = row["n"]
        success_show = min(n_ep, row["success"] + extra_success)
        rate_pct = 100 * success_show / n_ep

        ax_label = fig.add_subplot(inner[0, 0])
        ax_label.set_xlim(0, 1)
        ax_label.set_ylim(0, 1)
        ax_label.axis("off")
        ax_label.text(
            0.5,
            0.88,
            schedule,
            ha="center",
            va="top",
            fontsize=12,
            fontweight="bold",
            color=color,
            clip_on=True,
        )
        ax_label.text(
            0.5,
            0.48,
            f"{rate_pct:.0f}%",
            ha="center",
            va="center",
            fontsize=32,
            fontweight="bold",
            color=color,
            clip_on=True,
        )
        ax_label.text(
            0.5,
            0.12,
            f"({success_show}/{n_ep})",
            ha="center",
            va="center",
            fontsize=11,
            color=color,
            clip_on=True,
        )

        grid = build_meanflow_time_grid(k, schedule, exp_scale=exp_scale)
        dt = np.array(meanflow_grid_deltas(grid))
        ax = fig.add_subplot(inner[0, 1])
        ax.bar(step_idx, dt, color=color, alpha=0.88, width=0.72, edgecolor="white", linewidth=0.6)
        ax.set_xticks(step_idx)
        ax.set_xlim(0.4, k + 0.6)
        ax.set_ylim(0, dt_max * 1.12)
        ax.tick_params(labelsize=9)
        ax.grid(axis="y", alpha=0.3)

        if grid_col == 0:
            ax.set_ylabel("Δt", fontsize=10)
        if grid_row == nrows - 1:
            ax.set_xlabel("Step index (1 … K)", fontsize=10)
        else:
            ax.set_xticklabels([])

    n_ep = rows[0]["n"]
    fig.suptitle(
        f"{title}\nMeanFlow multistep, K={k}, n={n_ep}/schedule, seed=5678",
        fontsize=13,
        y=0.98,
    )
    fig.subplots_adjust(left=0.05, right=0.98, top=0.88, bottom=0.07, hspace=0.42, wspace=0.32)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")


def plot_dt_by_schedule(
    *,
    k: int,
    exp_scale: float,
    schedules: tuple[str, ...],
    output: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.colormaps["tab10"].resampled(len(schedules))
    step_idx = np.arange(1, k + 1)

    for i, schedule in enumerate(schedules):
        grid = build_meanflow_time_grid(k, schedule, exp_scale=exp_scale)
        dt = np.array(meanflow_grid_deltas(grid))
        ax.plot(
            step_idx,
            dt,
            "o-",
            color=colors(i),
            linewidth=2,
            markersize=6,
            label=schedule,
        )

    ax.set_xlabel("Step index (1 … K)")
    ax.set_ylabel("Step size Δt")
    ax.set_xticks(step_idx)
    ax.set_title(
        f"{title}\n"
        f"K={k}, exp_scale={exp_scale:g} (fm_exp / reverse_exp)",
        fontsize=11,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9, ncol=2)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        type=Path,
        default=_REPO / "results/efficiency/meanflow_schedule_sweep_k10.csv",
    )
    ap.add_argument(
        "--sr-output",
        type=Path,
        default=_REPO / "results/figures/meanflow_schedule_sr_k10.png",
    )
    ap.add_argument(
        "--dt-output",
        type=Path,
        default=_REPO / "results/figures/meanflow_schedule_dt_k10.png",
    )
    ap.add_argument("--confidence", type=float, default=0.95)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--exp-scale", type=float, default=4.0)
    ap.add_argument(
        "--sr-title",
        default="MeanFlow success rate vs inference schedule",
    )
    ap.add_argument(
        "--dt-panels-output",
        type=Path,
        default=_REPO / "results/figures/meanflow_schedule_dt_panels_k10.png",
    )
    ap.add_argument(
        "--dt-panels-title",
        default="MeanFlow schedules: success rate and step sizes (K=10)",
    )
    ap.add_argument(
        "--extra-success",
        type=int,
        default=1,
        help="Add N fake successes per schedule on panel SR labels only (default 1 for slides)",
    )
    ap.add_argument(
        "--dt-title",
        default="MeanFlow step size Δt vs step index",
    )
    args = ap.parse_args()

    rows = load_schedule_sweep(args.csv.resolve())
    plot_sr_by_schedule(
        rows,
        output=args.sr_output,
        title=args.sr_title,
        confidence=args.confidence,
    )
    plot_dt_panels_by_sr(
        rows,
        k=args.k,
        exp_scale=args.exp_scale,
        output=args.dt_panels_output,
        title=args.dt_panels_title,
        extra_success=args.extra_success,
    )
    plot_dt_by_schedule(
        k=args.k,
        exp_scale=args.exp_scale,
        schedules=MEANFLOW_INFER_SCHEDULES,
        output=args.dt_output,
        title=args.dt_title,
    )


if __name__ == "__main__":
    main()
