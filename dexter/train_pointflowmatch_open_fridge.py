import argparse
import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from dexter._paths import pythonpath_env, repo_root


def main() -> None:
    """
    Launcher for PointFlowMatch training via scripts/train.py (Hydra).

    Examples:
        # Baseline FMPolicy
        python dexter/train_pointflowmatch_open_fridge.py

        # Oracle phase conditioning (GT phases in dataset)
        python dexter/train_pointflowmatch_open_fridge.py --phase-conditioning enabled

        # Learned phase prediction (FMPolicy + phase_head)
        python dexter/train_pointflowmatch_open_fridge.py \\
            --phase-conditioning enabled --phase-prediction enabled

        # Gripper-weighted + learned phase
        python dexter/train_pointflowmatch_open_fridge.py \\
            --experiment pointflowmatch_gripper_weighted \\
            --phase-conditioning enabled --phase-prediction enabled
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
        help="Experiment from conf/experiment (pointflowmatch, pointflowmatch_gripper_weighted, ...).",
    )
    parser.add_argument(
        "--phase-conditioning",
        type=str,
        default="disabled",
        choices=["disabled", "enabled", "on", "off"],
        help="Hydra config group phase_conditioning (enabled/on = GT phase labels for flow).",
    )
    parser.add_argument(
        "--phase-prediction",
        type=str,
        default="disabled",
        choices=["disabled", "enabled"],
        help="Hydra config group phase_prediction (enabled = phase_head + learned phase at infer).",
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
        "--batch-size",
        type=int,
        default=None,
        help="Override dataloader.batch_size (e.g. 64 if OOM).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override dataloader.num_workers.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only run dexter/verify_training_setup.py with the same overrides, then exit.",
    )
    parser.add_argument(
        "--log-wandb",
        action="store_true",
        help="Enable online logging to Weights & Biases.",
    )
    parser.add_argument(
        "--checkpoint-schedule",
        type=str,
        default="milestones_1500",
        help="Hydra group checkpoint_schedule (milestones_1500 = ep 300,600,900,1200,1500).",
    )
    args = parser.parse_args()

    pc = args.phase_conditioning
    if pc == "on":
        pc = "enabled"
    elif pc == "off":
        pc = "disabled"

    repo = repo_root()
    train_script = repo / "scripts" / "train.py"
    verify_script = repo / "dexter" / "verify_training_setup.py"
    env = pythonpath_env(repo)

    if not train_script.exists():
        raise FileNotFoundError(f"Could not find training script at {train_script}")

    overrides = [
        f"task_name={args.task}",
        f"+experiment={args.experiment}",
        f"checkpoint_schedule={args.checkpoint_schedule}",
        f"phase_conditioning={pc}",
        f"phase_prediction={args.phase_prediction}",
        "launch_eval_after_train=false",
    ]
    if args.epochs is not None:
        overrides.append(f"epochs={args.epochs}")
    if args.run_name is not None:
        overrides.append(f"run_name={args.run_name}")
    if args.batch_size is not None:
        overrides.append(f"dataloader.batch_size={args.batch_size}")
    if args.num_workers is not None:
        overrides.append(f"dataloader.num_workers={args.num_workers}")
    if args.log_wandb:
        overrides.append("log_wandb=True")

    if args.verify_only:
        cmd = [sys.executable, str(verify_script), "--overrides", *overrides]
        print("Running verify:")
        print("  " + " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=repo, env=env)
        return

    cmd_verify = [sys.executable, str(verify_script), "--overrides", *overrides]
    subprocess.run(cmd_verify, check=True, cwd=repo, env=env)

    cmd = [sys.executable, str(train_script), *overrides]
    print("Running training command:")
    print("  " + " ".join(cmd))
    if env.get("PYTHONPATH"):
        print(f"  PYTHONPATH={env['PYTHONPATH']}")
    subprocess.run(cmd, check=True, cwd=repo, env=env)


if __name__ == "__main__":
    main()
