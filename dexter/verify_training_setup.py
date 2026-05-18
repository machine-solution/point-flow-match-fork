#!/usr/bin/env python3
"""
Verify Hydra training config instantiates FMPolicy with expected phase settings.

Run from repo root (no GPU / dataset required). Needs diffusion_policy on PYTHONPATH
(sibling ../diffusion_policy or pip install -e ../diffusion_policy), same as training.

  cd ~/point_flow_match/PointFlowMatch
  conda activate ./pfp-train-env
  python dexter/verify_training_setup.py --overrides task_name=open_fridge +experiment=pointflowmatch phase_conditioning=enabled phase_prediction=enabled
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Repo root on sys.path so `dexter._paths` and `pfp` import work when invoked as a script.
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from dexter._paths import repo_root, require_diffusion_policy, setup_training_pythonpath

setup_training_pythonpath(_REPO)
require_diffusion_policy(_REPO)

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-dir", type=Path, default=None)
    ap.add_argument(
        "--overrides",
        nargs="*",
        default=[
            "task_name=open_fridge",
            "+experiment=pointflowmatch",
            "phase_conditioning=disabled",
            "phase_prediction=disabled",
        ],
        help="Hydra overrides (same as scripts/train.py CLI)",
    )
    args = ap.parse_args()

    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)

    repo = repo_root()
    conf_dir = args.config_dir or (repo / "conf")
    if not conf_dir.is_dir():
        print(f"ERROR: config dir not found: {conf_dir}", file=sys.stderr)
        sys.exit(1)

    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg = compose(config_name="train", overrides=list(args.overrides))
    OmegaConf.resolve(cfg)

    target = str(cfg.model._target_)
    if not target.endswith("FMPolicy"):
        print(f"ERROR: expected FMPolicy, got {target}", file=sys.stderr)
        sys.exit(1)

    pcfg = getattr(cfg, "phase_conditioning", None)
    ppred = getattr(cfg, "phase_prediction", None)
    phase_on = bool(getattr(pcfg, "enabled", False)) if pcfg is not None else False
    pred_on = bool(getattr(ppred, "enabled", False)) if ppred is not None else False

    model = instantiate(
        cfg.model,
        phase_conditioning=pcfg,
        phase_prediction=ppred,
        phase_rollout=getattr(cfg, "phase_rollout", None),
    )

    print("OK verify_training_setup")
    print(f"  model._target_: {target}")
    print(f"  task_name: {cfg.task_name}")
    print(f"  experiment overrides: {args.overrides}")
    print(f"  phase_conditioning.enabled: {phase_on}")
    print(f"  phase_prediction.enabled: {pred_on}")
    print(f"  model.phase_enabled: {getattr(model, 'phase_enabled', '?')}")
    print(f"  model.phase_pred_enabled: {getattr(model, 'phase_pred_enabled', '?')}")
    print(f"  model.phase_head: {'present' if getattr(model, 'phase_head', None) is not None else 'None'}")

    if pred_on and not phase_on:
        print("ERROR: phase_prediction without phase_conditioning", file=sys.stderr)
        sys.exit(1)
    if pred_on and getattr(model, "phase_head", None) is None:
        print("ERROR: phase_prediction enabled but phase_head is None", file=sys.stderr)
        sys.exit(1)
    if phase_on and getattr(model, "phase_embedding", None) is None:
        print("ERROR: phase_conditioning enabled but phase_embedding is None", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
