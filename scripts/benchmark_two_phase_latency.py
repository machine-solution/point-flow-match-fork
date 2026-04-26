"""
Micro-benchmark: latency of one policy.predict_action() call.

Compares:
  1) baseline single-model policy
  2) two-phase wrapper policy (pre+post), staying in pre phase

Run (from repo root, in an env with torch + model deps):
  python scripts/benchmark_two_phase_latency.py

Optional overrides:
  BASELINE_CKPT=open_fridge_0803_modify PRE_CKPT=open_fridge_pre_891 POST_CKPT=open_fridge_post_953 python ...
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass

import numpy as np

# Add diffusion_policy to path if not already there
_dp = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "diffusion_policy")
if os.path.exists(_dp) and _dp not in sys.path:
    sys.path.insert(0, _dp)

import torch
import hydra
from omegaconf import OmegaConf

from pfp import REPO_DIRS
from pfp.policy.base_policy import BasePolicy
from pfp.policy.two_phase_policy import TwoPhasePolicy


@dataclass(frozen=True)
class BenchResult:
    name: str
    device: str
    iters: int
    mean_ms: float
    p50_ms: float
    p90_ms: float
    p99_ms: float


def _load_policy(ckpt_name: str, *, ckpt_episode: str = "latest") -> BasePolicy:
    ckpt_path = REPO_DIRS.CKPT / ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    train_cfg = OmegaConf.load(ckpt_path / "config.yaml")
    policy_class = hydra.utils.get_class(train_cfg.model._target_)
    policy: BasePolicy = policy_class.load_from_checkpoint(
        ckpt_name=ckpt_name,
        ckpt_episode=ckpt_episode,
        num_k_infer=int(getattr(train_cfg.model, "num_k_infer", 50)),
        flow_schedule=getattr(train_cfg.model, "flow_schedule", None),
        exp_scale=getattr(train_cfg.model, "exp_scale", None),
        subs_factor=getattr(train_cfg.model, "subs_factor", 1),
    )
    policy.reset_obs()
    return policy


def _bench(policy: BasePolicy, *, obs: np.ndarray, robot_state: np.ndarray, iters: int, warmup: int, name: str) -> BenchResult:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Warmup
    policy.reset_obs()
    for _ in range(warmup):
        _ = policy.predict_action(obs, robot_state)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Timed
    times = np.zeros((iters,), dtype=np.float64)
    policy.reset_obs()
    for i in range(iters):
        t0 = time.perf_counter()
        _ = policy.predict_action(obs, robot_state)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times[i] = (time.perf_counter() - t0) * 1000.0

    return BenchResult(
        name=name,
        device=device,
        iters=iters,
        mean_ms=float(times.mean()),
        p50_ms=float(np.percentile(times, 50)),
        p90_ms=float(np.percentile(times, 90)),
        p99_ms=float(np.percentile(times, 99)),
    )


def main() -> None:
    baseline_ckpt = os.environ.get("BASELINE_CKPT", "open_fridge_0803_modify")
    pre_ckpt = os.environ.get("PRE_CKPT", "open_fridge_pre_891")
    post_ckpt = os.environ.get("POST_CKPT", "open_fridge_post_953")
    iters = int(os.environ.get("ITERS", "50"))
    warmup = int(os.environ.get("WARMUP", "5"))

    # Use n_points from baseline config if present; otherwise fall back to 4096.
    n_points = 4096
    try:
        train_cfg = OmegaConf.load((REPO_DIRS.CKPT / baseline_ckpt / "config.yaml"))
        n_points = int(getattr(getattr(train_cfg, "dataset", {}), "n_points", 4096))
    except Exception:
        pass

    obs = (np.random.rand(n_points, 3).astype(np.float32) - 0.5).astype(np.float32)
    robot_state = np.zeros((10,), dtype=np.float32)
    robot_state[9] = 1.0  # open gripper -> should stay in pre for TwoPhasePolicy

    print(f"[bench] device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"[bench] baseline={baseline_ckpt}  pre={pre_ckpt}  post={post_ckpt}")
    print(f"[bench] n_points={n_points} warmup={warmup} iters={iters}")

    baseline = _load_policy(baseline_ckpt, ckpt_episode="latest")
    pre = _load_policy(pre_ckpt, ckpt_episode="latest")
    post = _load_policy(post_ckpt, ckpt_episode="latest")
    two_phase = TwoPhasePolicy(pre, post, gripper_thr=0.5, closed_steps_to_switch=3)

    r1 = _bench(baseline, obs=obs, robot_state=robot_state, iters=iters, warmup=warmup, name="single")
    r2 = _bench(two_phase, obs=obs, robot_state=robot_state, iters=iters, warmup=warmup, name="two_phase(pre)")

    def _fmt(r: BenchResult) -> str:
        return (
            f"{r.name:14s}  mean={r.mean_ms:7.1f} ms  "
            f"p50={r.p50_ms:7.1f}  p90={r.p90_ms:7.1f}  p99={r.p99_ms:7.1f}  ({r.iters} iters, {r.device})"
        )

    print(_fmt(r1))
    print(_fmt(r2))
    print(f"ratio(two_phase/single) = {r2.mean_ms / max(r1.mean_ms, 1e-9):.2f}x")


if __name__ == "__main__":
    main()

