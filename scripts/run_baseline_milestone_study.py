#!/usr/bin/env python3
"""Run milestone validation sweep and aggregate JSON/CSV for plotting."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_MILESTONES = (300, 600, 900, 1200, 1500)
MILESTONES_DESC = (1500, 1200, 900, 600, 300)


def _parse_milestones(s: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-name", default="1779122560-baseline-many-ckpts")
    ap.add_argument("--num-episodes", type=int, default=100)
    ap.add_argument(
        "--max-episode-length",
        type=int,
        default=200,
        help="RLBench rollout cap (eval.yaml default 200). Use 120 if successes are always <110 steps.",
    )
    ap.add_argument("--seed", type=int, default=5678)
    ap.add_argument(
        "--milestones",
        type=str,
        default=None,
        help="Comma-separated train epochs, e.g. 1500,1200,900,600,300",
    )
    ap.add_argument(
        "--order",
        choices=("asc", "desc"),
        default="desc",
        help="If --milestones omitted: asc=300..1500, desc=1500..300 (default desc).",
    )
    ap.add_argument("--conda-env", default="pfp_env")
    ap.add_argument("--repo-root", type=Path, default=None)
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip milestone if ep{N}.json already exists in output dir.",
    )
    ap.add_argument(
        "--phase-conditioning",
        choices=("enabled", "disabled"),
        default="disabled",
    )
    ap.add_argument(
        "--phase-prediction",
        choices=("enabled", "disabled"),
        default="disabled",
    )
    args = ap.parse_args()

    if args.milestones:
        milestones = _parse_milestones(args.milestones)
    elif args.order == "desc":
        milestones = MILESTONES_DESC
    else:
        milestones = DEFAULT_MILESTONES

    repo = (args.repo_root or Path(__file__).resolve().parents[1]).resolve()
    max_suffix = (
        f"_max{args.max_episode_length}" if args.max_episode_length != 200 else ""
    )
    out_dir = (
        repo
        / "results"
        / f"milestone_study_{args.ckpt_name}_seed{args.seed}_n{args.num_episodes}{max_suffix}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    cs41 = repo / "CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
    if cs41.is_dir():
        env["COPPELIASIM_ROOT"] = str(cs41)
        env["LD_LIBRARY_PATH"] = f"{cs41}:{env.get('LD_LIBRARY_PATH', '')}"
        env["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(cs41)
    dp = repo.parent / "diffusion_policy"
    if dp.is_dir():
        env["PYTHONPATH"] = f"{dp}:{repo}:{env.get('PYTHONPATH', '')}"
    env["PYTHONUNBUFFERED"] = "1"

    meta = {
        "ckpt_name": args.ckpt_name,
        "seed": args.seed,
        "num_episodes": args.num_episodes,
        "max_episode_length": args.max_episode_length,
        "milestones": list(milestones),
        "milestone_order": args.order,
        "phase_conditioning": args.phase_conditioning,
        "phase_prediction": args.phase_prediction,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runs": [],
    }

    print(f"Output: {out_dir}")
    print(
        f"Milestones (order): {list(milestones)}  seed={args.seed}  "
        f"n={args.num_episodes}  max_steps={args.max_episode_length}"
    )

    for ep in milestones:
        json_path = out_dir / f"ep{ep}.json"
        log_path = out_dir / f"ep{ep}.log"
        if args.skip_existing and json_path.is_file():
            print(f"\n>>> ep{ep} skip (exists {json_path})")
            run_data = json.loads(json_path.read_text(encoding="utf-8"))
            run_data["train_epoch"] = ep
            meta["runs"].append(run_data)
            continue
        cmd = [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            args.conda_env,
            "python",
            str(repo / "scripts" / "validate_accuracy.py"),
            f"policy.ckpt_name={args.ckpt_name}",
            f"policy.ckpt_episode=ep{ep}",
            f"env_runner.num_episodes={args.num_episodes}",
            f"env_runner.max_episode_length={args.max_episode_length}",
            f"seed={args.seed}",
            f"phase_conditioning={args.phase_conditioning}",
            f"phase_prediction={args.phase_prediction}",
            "env_runner.verbose=True",
            f"results_json={json_path}",
        ]
        print(f"\n{'=' * 60}\n>>> epoch {ep}\n{'=' * 60}")
        with log_path.open("w", encoding="utf-8") as logf:
            proc = subprocess.run(cmd, cwd=repo, env=env, stdout=logf, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            print(f"FAILED ep{ep} (exit {proc.returncode}), see {log_path}", file=sys.stderr)
            sys.exit(proc.returncode)
        run_data = json.loads(json_path.read_text(encoding="utf-8"))
        run_data["train_epoch"] = ep
        meta["runs"].append(run_data)
        print(f"ep{ep}: {run_data['num_success']}/{run_data['num_episodes']} ({100*run_data['accuracy']:.1f}%)")

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    csv_path = out_dir / "accuracy_by_epoch.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "train_epoch",
                "ckpt_episode",
                "seed",
                "num_episodes",
                "num_success",
                "accuracy",
                "avg_steps_successful",
            ],
        )
        w.writeheader()
        for r in meta["runs"]:
            w.writerow(
                {
                    "train_epoch": r["train_epoch"],
                    "ckpt_episode": r["ckpt_episode"],
                    "seed": r["seed"],
                    "num_episodes": r["num_episodes"],
                    "num_success": r["num_success"],
                    "accuracy": r["accuracy"],
                    "avg_steps_successful": r.get("avg_steps_successful"),
                }
            )

    episodes_csv = out_dir / "episodes_detail.csv"
    with episodes_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["train_epoch", "episode_idx", "success", "steps"]
        )
        w.writeheader()
        for r in meta["runs"]:
            te = r["train_epoch"]
            for ep in r["episodes"]:
                w.writerow(
                    {
                        "train_epoch": te,
                        "episode_idx": ep["episode_idx"],
                        "success": int(ep["success"]),
                        "steps": ep["steps"],
                    }
                )

    print(f"\nWrote:\n  {summary_path}\n  {csv_path}\n  {episodes_csv}\n  per-epoch JSON in {out_dir}")


if __name__ == "__main__":
    main()
