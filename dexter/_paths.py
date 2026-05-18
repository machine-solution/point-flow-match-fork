"""Shared repo paths for Dexter launchers (matches sbatch PYTHONPATH)."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def diffusion_policy_dir(repo: Path | None = None) -> Path:
    r = repo or repo_root()
    return r.parent / "diffusion_policy"


def setup_training_pythonpath(repo: Path | None = None) -> list[str]:
    """
    Prepend sibling diffusion_policy and repo root to sys.path / PYTHONPATH.

    Same layout as dexter/run_pointflowmatch_open_fridge.sbatch:
      export PYTHONPATH="${SLURM_SUBMIT_DIR}/../diffusion_policy:..."
    """
    r = (repo or repo_root()).resolve()
    added: list[str] = []
    dp = diffusion_policy_dir(r)
    if dp.is_dir():
        added.append(str(dp.resolve()))
    added.append(str(r))
    for p in reversed(added):
        if p not in sys.path:
            sys.path.insert(0, p)
    return added


def pythonpath_env(repo: Path | None = None) -> dict[str, str]:
    """os.environ copy with PYTHONPATH updated for subprocess calls."""
    env = os.environ.copy()
    prefix = os.pathsep.join(setup_training_pythonpath(repo))
    old = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{prefix}{os.pathsep}{old}" if old else prefix
    return env


def require_diffusion_policy(repo: Path | None = None) -> None:
    """Exit with a clear message if diffusion_policy cannot be imported."""
    setup_training_pythonpath(repo)
    try:
        import diffusion_policy  # noqa: F401
    except ImportError:
        r = repo or repo_root()
        dp = diffusion_policy_dir(r)
        print(
            "ERROR: cannot import diffusion_policy.\n\n"
            "On Dexter this repo must sit next to diffusion_policy, e.g.:\n"
            "  ~/point_flow_match/PointFlowMatch\n"
            "  ~/point_flow_match/diffusion_policy\n\n"
            "Fix (from PointFlowMatch root, with conda env active):\n"
            f"  export PYTHONPATH=\"{dp.resolve()}:{r.resolve()}:$PYTHONPATH\"\n"
            "  pip install -e ../diffusion_policy\n"
            "  pip install -e . --no-deps\n",
            file=sys.stderr,
        )
        if not dp.is_dir():
            print(f"  (missing directory: {dp})", file=sys.stderr)
        sys.exit(1)
