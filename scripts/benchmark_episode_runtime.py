"""
Benchmark wall-clock time per *episode* (full rollout) in RLBench/CoppeliaSim.

This measures what you actually care about: how long it takes to generate actions
AND step the simulator until success/terminate/max_episode_length.

Compares:
  - single baseline policy (one model)
  - two-phase policy (pre->post switch by observed gripper)

Run (from repo root, in an env with RLBench working):
  xvfb-run -a python scripts/benchmark_episode_runtime.py

Overrides via env:
  BASELINE_CKPT=open_fridge_0803_modify PRE_CKPT=open_fridge_pre_891 POST_CKPT=open_fridge_post_953
  TASK=open_fridge EPISODES=10 MAX_STEPS=200 BASE_SEED=5678
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

import hydra
from omegaconf import OmegaConf

from pfp import REPO_DIRS, set_seeds
from pfp.envs.rlbench_env import RLBenchEnv
from pfp.policy.base_policy import BasePolicy
from pfp.policy.two_phase_policy import TwoPhasePolicy


@dataclass(frozen=True)
class EpisodeStat:
    wall_s: float
    steps: int
    success: bool


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
    return policy


def _rollout(
    *,
    policy: BasePolicy,
    env: RLBenchEnv,
    episode_seed: int,
    max_steps: int,
) -> EpisodeStat:
    set_seeds(episode_seed)
    policy.reset_obs()
    env.reset()

    t0 = time.perf_counter()
    success = False
    steps = 0
    for step in range(max_steps):
        robot_state, obs = env.get_obs()
        if hasattr(policy, "set_step"):
            try:
                policy.set_step(step)  # type: ignore[attr-defined]
            except Exception:
                pass
        pred = policy.predict_action(obs, robot_state)
        next_robot_state = pred[-1, 0]
        reward, terminate = env.step(next_robot_state)
        success = bool(reward)
        steps = step + 1
        if success or terminate:
            break
    wall = time.perf_counter() - t0
    return EpisodeStat(wall_s=wall, steps=steps, success=success)


def _summarize(name: str, stats: list[EpisodeStat]) -> None:
    walls = np.array([s.wall_s for s in stats], dtype=np.float64)
    steps = np.array([s.steps for s in stats], dtype=np.int32)
    succ = np.array([s.success for s in stats], dtype=np.int32)
    n = len(stats)
    print(f"\n== {name} ==")
    print(f"episodes: {n}  success: {succ.sum()}/{n} ({100.0 * succ.mean():.1f}%)")
    print(
        f"wall_s: mean={walls.mean():.2f}  p50={np.percentile(walls, 50):.2f}  "
        f"p90={np.percentile(walls, 90):.2f}  max={walls.max():.2f}"
    )
    print(
        f"steps:  mean={steps.mean():.1f}  p50={np.percentile(steps, 50):.0f}  "
        f"p90={np.percentile(steps, 90):.0f}  max={steps.max():.0f}"
    )


def main() -> None:
    baseline_ckpt = os.environ.get("BASELINE_CKPT", "open_fridge_0803_modify")
    pre_ckpt = os.environ.get("PRE_CKPT", "open_fridge_pre_891")
    post_ckpt = os.environ.get("POST_CKPT", "open_fridge_post_953")

    task = os.environ.get("TASK", "open_fridge")
    episodes = int(os.environ.get("EPISODES", "5"))
    max_steps = int(os.environ.get("MAX_STEPS", "200"))
    base_seed = int(os.environ.get("BASE_SEED", "5678"))

    # Env settings (match eval defaults)
    env_cfg = dict(
        task_name=task,
        voxel_size=0.01,
        n_points=4096,
        use_pc_color=False,
        headless=True,
        vis=False,
        obs_mode="pcd",
    )

    print("[bench-episode] baseline:", baseline_ckpt)
    print("[bench-episode] two-phase pre/post:", pre_ckpt, post_ckpt)
    print(f"[bench-episode] task={task} episodes={episodes} max_steps={max_steps} base_seed={base_seed}")

    # Load policies
    baseline = _load_policy(baseline_ckpt)
    pre = _load_policy(pre_ckpt)
    post = _load_policy(post_ckpt)
    two_phase = TwoPhasePolicy(pre, post, gripper_thr=0.5, closed_steps_to_switch=3)

    # For fairness, run both with identical per-episode seeds.
    # IMPORTANT: RLBench/CoppeliaSim is not happy with multiple Environment() launches
    # in a single process. So we use ONE env instance and just switch the policy.
    seeds = [base_seed + i for i in range(episodes)]

    env = RLBenchEnv(**env_cfg)
    stats1 = [_rollout(policy=baseline, env=env, episode_seed=s, max_steps=max_steps) for s in seeds]
    _summarize("single", stats1)

    stats2 = [_rollout(policy=two_phase, env=env, episode_seed=s, max_steps=max_steps) for s in seeds]
    _summarize("two_phase", stats2)

    # Ratio (mean wall time)
    r = (np.mean([s.wall_s for s in stats2]) / max(np.mean([s.wall_s for s in stats1]), 1e-9))
    print(f"\nratio(two_phase/single) mean wall time = {r:.2f}x")


if __name__ == "__main__":
    main()

