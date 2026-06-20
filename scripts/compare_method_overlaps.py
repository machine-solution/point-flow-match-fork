#!/usr/bin/env python3
"""Run per-episode evals and compute differential overlap analysis across methods."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "comparison"


@dataclass
class MethodSpec:
    name: str
    ckpt_name: str
    ckpt_episode: str
    k: int
    meanflow_multistep: bool = False


def _run_validate(spec: MethodSpec, *, seed: int, num_episodes: int, max_episode_length: int, out_json: Path) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "validate_accuracy.py"),
        f"policy.ckpt_name={spec.ckpt_name}",
        f"policy.ckpt_episode={spec.ckpt_episode}",
        f"policy.num_k_infer={int(spec.k)}",
        f"seed={int(seed)}",
        f"env_runner.num_episodes={int(num_episodes)}",
        f"env_runner.max_episode_length={int(max_episode_length)}",
        "env_runner.verbose=false",
        f"results_json={str(out_json)}",
    ]
    if spec.meanflow_multistep:
        # policy.meanflow_multistep_infer does not exist in eval config schema by default.
        cmd.append("+policy.meanflow_multistep_infer=true")
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"[{spec.name}] validate_accuracy failed (exit={proc.returncode})\n"
            f"--- stdout ---\n{proc.stdout[-6000:]}\n"
            f"--- stderr ---\n{proc.stderr[-6000:]}"
        )
    if not out_json.exists():
        raise FileNotFoundError(f"[{spec.name}] expected results JSON not found: {out_json}")


def _load_episode_success(path: Path, *, method_name: str, num_episodes: int) -> list[int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    episodes = data.get("episodes", [])
    if len(episodes) != num_episodes:
        raise ValueError(f"[{method_name}] expected {num_episodes} episodes, got {len(episodes)}")
    by_idx: dict[int, int] = {}
    for row in episodes:
        idx = int(row["episode_idx"])
        by_idx[idx] = 1 if bool(row["success"]) else 0
    missing = [i for i in range(num_episodes) if i not in by_idx]
    if missing:
        raise ValueError(f"[{method_name}] missing episode indices: {missing[:10]}")
    return [by_idx[i] for i in range(num_episodes)]


def _write_episode_matrix(out_csv: Path, *, fm: list[int], shortcut: list[int], meanflow: list[int]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["episode_id", "fm", "shortcut", "meanflow"])
        w.writeheader()
        for i, (a, b, c) in enumerate(zip(fm, shortcut, meanflow)):
            w.writerow({"episode_id": i, "fm": a, "shortcut": b, "meanflow": c})


def _cond_prob(num: int, den: int) -> float | None:
    if den <= 0:
        return None
    return float(num / den)


def _fmt_prob(p: float | None) -> str:
    if p is None:
        return "n/a"
    return f"{100.0 * p:.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=5678)
    ap.add_argument("--num-episodes", type=int, default=100)
    ap.add_argument("--max-episode-length", type=int, default=120)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--reuse-existing", action="store_true")
    ap.add_argument("--baseline-ckpt", default="1779122560-baseline-many-ckpts")
    ap.add_argument("--baseline-episode", default="latest")
    ap.add_argument("--baseline-k", type=int, default=10)
    ap.add_argument("--shortcut-ckpt", default="shortcut_open_fridge_1385")
    ap.add_argument("--shortcut-episode", default="latest")
    ap.add_argument("--shortcut-k", type=int, default=10)
    ap.add_argument("--meanflow-ckpt", default="meanflow_open_fridge_1365")
    ap.add_argument("--meanflow-episode", default="latest")
    ap.add_argument("--meanflow-k", type=int, default=1)
    ap.add_argument(
        "--meanflow-multistep",
        action="store_true",
        help="Enable true K-step MeanFlow sampler (+policy.meanflow_multistep_infer=true).",
    )
    args = ap.parse_args()

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        MethodSpec(
            name="fm",
            ckpt_name=args.baseline_ckpt,
            ckpt_episode=args.baseline_episode,
            k=int(args.baseline_k),
            meanflow_multistep=False,
        ),
        MethodSpec(
            name="shortcut",
            ckpt_name=args.shortcut_ckpt,
            ckpt_episode=args.shortcut_episode,
            k=int(args.shortcut_k),
            meanflow_multistep=False,
        ),
        MethodSpec(
            name="meanflow",
            ckpt_name=args.meanflow_ckpt,
            ckpt_episode=args.meanflow_episode,
            k=int(args.meanflow_k),
            meanflow_multistep=bool(args.meanflow_multistep),
        ),
    ]

    run_json_paths: dict[str, Path] = {}
    for spec in specs:
        p = out_dir / f"{spec.name}_episodes.json"
        run_json_paths[spec.name] = p
        if args.reuse_existing and p.exists():
            print(f"[reuse] {spec.name}: {p}")
            continue
        print(
            f"[run] {spec.name}: ckpt={spec.ckpt_name} episode={spec.ckpt_episode} "
            f"K={spec.k} meanflow_multistep={spec.meanflow_multistep}"
        )
        _run_validate(
            spec,
            seed=int(args.seed),
            num_episodes=int(args.num_episodes),
            max_episode_length=int(args.max_episode_length),
            out_json=p,
        )

    fm = _load_episode_success(run_json_paths["fm"], method_name="fm", num_episodes=int(args.num_episodes))
    shortcut = _load_episode_success(
        run_json_paths["shortcut"], method_name="shortcut", num_episodes=int(args.num_episodes)
    )
    meanflow = _load_episode_success(
        run_json_paths["meanflow"], method_name="meanflow", num_episodes=int(args.num_episodes)
    )
    n = int(args.num_episodes)

    matrix_csv = out_dir / "episode_matrix.csv"
    _write_episode_matrix(matrix_csv, fm=fm, shortcut=shortcut, meanflow=meanflow)

    # Set-based stats.
    fm_set = {i for i, v in enumerate(fm) if v == 1}
    sc_set = {i for i, v in enumerate(shortcut) if v == 1}
    mf_set = {i for i, v in enumerate(meanflow) if v == 1}
    all_idx = set(range(n))

    combo_to_ids: dict[str, list[int]] = {f"{a}{b}{c}": [] for a in (0, 1) for b in (0, 1) for c in (0, 1)}
    for i in range(n):
        key = f"{fm[i]}{shortcut[i]}{meanflow[i]}"
        combo_to_ids[key].append(i)

    overlap = {
        "n_episodes": n,
        "counts": {
            "fm_success": len(fm_set),
            "shortcut_success": len(sc_set),
            "meanflow_success": len(mf_set),
            "fm_inter_shortcut": len(fm_set & sc_set),
            "fm_inter_meanflow": len(fm_set & mf_set),
            "shortcut_inter_meanflow": len(sc_set & mf_set),
            "all_three": len(fm_set & sc_set & mf_set),
            "fm_only": len(fm_set - sc_set - mf_set),
            "shortcut_only": len(sc_set - fm_set - mf_set),
            "meanflow_only": len(mf_set - fm_set - sc_set),
        },
        "conditional_probabilities": {
            "P(FM|Shortcut)": _cond_prob(len(fm_set & sc_set), len(sc_set)),
            "P(Shortcut|FM)": _cond_prob(len(fm_set & sc_set), len(fm_set)),
            "P(FM|MeanFlow)": _cond_prob(len(fm_set & mf_set), len(mf_set)),
            "P(MeanFlow|FM)": _cond_prob(len(fm_set & mf_set), len(fm_set)),
            "P(Shortcut|MeanFlow)": _cond_prob(len(sc_set & mf_set), len(mf_set)),
            "P(MeanFlow|Shortcut)": _cond_prob(len(sc_set & mf_set), len(sc_set)),
        },
        "combination_counts": {k: len(v) for k, v in combo_to_ids.items()},
        "episode_groups": {
            "fm_only": sorted(list(fm_set - sc_set - mf_set)),
            "shortcut_only": sorted(list(sc_set - fm_set - mf_set)),
            "meanflow_only": sorted(list(mf_set - fm_set - sc_set)),
            "all_success": sorted(list(fm_set & sc_set & mf_set)),
            "all_fail": sorted(list(all_idx - (fm_set | sc_set | mf_set))),
            "fm_shortcut_only": sorted(list((fm_set & sc_set) - mf_set)),
            "fm_meanflow_only": sorted(list((fm_set & mf_set) - sc_set)),
            "shortcut_meanflow_only": sorted(list((sc_set & mf_set) - fm_set)),
        },
        "runs": {
            spec.name: {
                "ckpt_name": spec.ckpt_name,
                "ckpt_episode": spec.ckpt_episode,
                "k": spec.k,
                "meanflow_multistep": spec.meanflow_multistep,
                "json_path": str(run_json_paths[spec.name]),
            }
            for spec in specs
        },
        "failure_reason_stats": {
            "available": False,
            "note": "Current validate_accuracy export does not include structured failure reasons "
            "(collision/timeout/IK/planner).",
        },
    }

    stats_json = out_dir / "overlap_stats.json"
    stats_json.write_text(json.dumps(overlap, indent=2), encoding="utf-8")

    groups_json = out_dir / "episode_groups.json"
    groups_json.write_text(json.dumps(overlap["episode_groups"], indent=2), encoding="utf-8")

    c = overlap["counts"]
    cp = overlap["conditional_probabilities"]
    report_md = out_dir / "overlap_report.md"
    report_md.write_text(
        "\n".join(
            [
                "# Differential Error Analysis",
                "",
                f"- Episodes: `{n}` (fixed `episode_id=0..{n-1}`, seed `{args.seed}`)",
                f"- Baseline FM: `{args.baseline_ckpt}` (K={args.baseline_k})",
                f"- Shortcut: `{args.shortcut_ckpt}` (K={args.shortcut_k})",
                f"- MeanFlow: `{args.meanflow_ckpt}` (K={args.meanflow_k}, multistep={bool(args.meanflow_multistep)})",
                "",
                "## Success Counts",
                f"- FM: `{c['fm_success']}`",
                f"- Shortcut: `{c['shortcut_success']}`",
                f"- MeanFlow: `{c['meanflow_success']}`",
                "",
                "## Intersections",
                f"- FM ∩ Shortcut: `{c['fm_inter_shortcut']}`",
                f"- FM ∩ MeanFlow: `{c['fm_inter_meanflow']}`",
                f"- Shortcut ∩ MeanFlow: `{c['shortcut_inter_meanflow']}`",
                f"- FM ∩ Shortcut ∩ MeanFlow: `{c['all_three']}`",
                "",
                "## Unique Successes",
                f"- FM only: `{c['fm_only']}`",
                f"- Shortcut only: `{c['shortcut_only']}`",
                f"- MeanFlow only: `{c['meanflow_only']}`",
                "",
                "## Conditional Probabilities",
                f"- P(FM success | Shortcut success): `{_fmt_prob(cp['P(FM|Shortcut)'])}`",
                f"- P(Shortcut success | FM success): `{_fmt_prob(cp['P(Shortcut|FM)'])}`",
                f"- P(FM success | MeanFlow success): `{_fmt_prob(cp['P(FM|MeanFlow)'])}`",
                f"- P(MeanFlow success | FM success): `{_fmt_prob(cp['P(MeanFlow|FM)'])}`",
                f"- P(Shortcut success | MeanFlow success): `{_fmt_prob(cp['P(Shortcut|MeanFlow)'])}`",
                f"- P(MeanFlow success | Shortcut success): `{_fmt_prob(cp['P(MeanFlow|Shortcut)'])}`",
                "",
                "## Full 3-bit Combination Counts",
                f"- 000: `{overlap['combination_counts']['000']}`",
                f"- 001: `{overlap['combination_counts']['001']}`",
                f"- 010: `{overlap['combination_counts']['010']}`",
                f"- 011: `{overlap['combination_counts']['011']}`",
                f"- 100: `{overlap['combination_counts']['100']}`",
                f"- 101: `{overlap['combination_counts']['101']}`",
                f"- 110: `{overlap['combination_counts']['110']}`",
                f"- 111: `{overlap['combination_counts']['111']}`",
                "",
                "## Key Observations (Auto)",
                f"- FM solves `{c['fm_only']}` episodes that both Shortcut and MeanFlow miss.",
                f"- Shortcut solves `{c['shortcut_only']}` episodes that both FM and MeanFlow miss.",
                f"- MeanFlow solves `{c['meanflow_only']}` episodes that both FM and Shortcut miss.",
                f"- Common successes across all three: `{c['all_three']}` episodes.",
                "",
                "Failure reason breakdown is unavailable in current exports; "
                "requires extending validate runner with structured failure codes.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Wrote: {matrix_csv}")
    print(f"Wrote: {stats_json}")
    print(f"Wrote: {report_md}")
    print(f"Wrote: {groups_json}")


if __name__ == "__main__":
    main()
