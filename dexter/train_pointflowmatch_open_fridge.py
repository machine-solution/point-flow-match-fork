import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    """
    Simple launcher for training a PointFlowMatch baseline on a single RLBench task.

    By default it starts training the PointFlowMatch baseline on the `open_fridge`
    task using the existing Hydra training pipeline in `scripts/train.py`.

    Example:
        python dexter/train_pointflowmatch_open_fridge.py

    You can override task, experiment, epochs and run name, e.g.:
        python dexter/train_pointflowmatch_open_fridge.py \\
            --task unplug_charger \\
            --experiment pointflowmatch \\
            --epochs 1500 \\
            --run-name my_unplug_run
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="open_fridge",
        help="RLBench task name (e.g. open_fridge, unplug_charger, close_door, ...).",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="pointflowmatch",
        help="Experiment config name from conf/experiment (e.g. pointflowmatch, dp3, adaflow).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides conf/train.yaml if set).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run name for checkpoints (maps to cfg.run_name).",
    )
    parser.add_argument(
        "--log-wandb",
        action="store_true",
        help="Enable online logging to Weights & Biases.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    train_script = repo_root / "scripts" / "train.py"

    if not train_script.exists():
        raise FileNotFoundError(f"Could not find training script at {train_script}")

    # Build Hydra command line
    cmd = [
        sys.executable,
        str(train_script),
        f"task_name={args.task}",
        f"+experiment={args.experiment}",
    ]

    # Explicit wandb mode if requested
    if args.log_wandb:
        cmd.append("log_wandb=True")

    if args.epochs is not None:
        cmd.append(f"epochs={args.epochs}")

    if args.run_name is not None:
        cmd.append(f"run_name={args.run_name}")

    print("Running training command:")
    print("  " + " ".join(cmd))

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

