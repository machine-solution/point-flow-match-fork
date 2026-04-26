"""
Валидация (N эпизодов) в симуляции, но с ДВУМЯ моделями: pre-grasp -> post-grasp.

Переключение (вариант A): по *наблюдаемому* gripper в robot_state:
  если robot_state[9] < thr `closed_steps_to_switch` шагов подряд -> post.

Пример:
  python scripts/validate_accuracy_two_phase.py \\
    policy_pre.ckpt_name=open_fridge_pre_1001 \\
    policy_post.ckpt_name=open_fridge_post_1002 \\
    env_runner.num_episodes=100
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

_diffusion_policy_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "..", "diffusion_policy"
)
if os.path.exists(_diffusion_policy_path) and _diffusion_policy_path not in sys.path:
    sys.path.insert(0, _diffusion_policy_path)

# PyTorch 2.7+: _refresh_per_optimizer_state в torch.amp.grad_scaler, Composer импортирует из torch.cuda.amp
import torch.cuda.amp.grad_scaler as _cuda_gs

if not hasattr(_cuda_gs, "_refresh_per_optimizer_state"):
    try:
        from torch.amp.grad_scaler import _refresh_per_optimizer_state

        _cuda_gs._refresh_per_optimizer_state = _refresh_per_optimizer_state
    except ImportError:
        pass

import hydra
import wandb
from omegaconf import OmegaConf, open_dict

from pfp import REPO_DIRS, set_seeds
from pfp.envs.rlbench_runner import RLBenchRunner
from pfp.policy.base_policy import BasePolicy
from pfp.policy.two_phase_policy import TwoPhasePolicy


def _load_policy_from_ckpt(cfg_policy, train_cfg) -> BasePolicy:
    policy_class = hydra.utils.get_class(train_cfg.model._target_)
    return policy_class.load_from_checkpoint(
        ckpt_name=cfg_policy.ckpt_name,
        ckpt_episode=cfg_policy.get("ckpt_episode", "latest"),
        num_k_infer=cfg_policy.get("num_k_infer", 50),
        flow_schedule=cfg_policy.get("flow_schedule", None),
        exp_scale=cfg_policy.get("exp_scale", None),
        subs_factor=cfg_policy.get("subs_factor", 1),
    )


@hydra.main(version_base=None, config_path="../conf", config_name="eval_two_phase")
def main(cfg: OmegaConf):
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    set_seeds(cfg.seed)

    # Runner logs metrics to wandb; keep it disabled by default.
    wandb.init(mode="disabled")

    ckpt_pre_path = REPO_DIRS.CKPT / cfg.policy_pre.ckpt_name
    ckpt_post_path = REPO_DIRS.CKPT / cfg.policy_post.ckpt_name
    if not ckpt_pre_path.exists():
        print(f"ERROR: pre checkpoint not found: {ckpt_pre_path}")
        return
    if not ckpt_post_path.exists():
        print(f"ERROR: post checkpoint not found: {ckpt_post_path}")
        return

    with open_dict(cfg):
        pre_train_cfg = OmegaConf.load(ckpt_pre_path / "config.yaml")
        post_train_cfg = OmegaConf.load(ckpt_post_path / "config.yaml")

        # Env config (берём из pre; предполагаем, что pre/post совместимы)
        cfg.env_runner.env_config.task_name = pre_train_cfg.task_name
        cfg.env_runner.env_config.obs_mode = pre_train_cfg.obs_mode
        cfg.env_runner.env_config.use_pc_color = pre_train_cfg.dataset.use_pc_color
        cfg.env_runner.env_config.n_points = pre_train_cfg.dataset.n_points
        cfg.env_runner.env_config.vis = False
        cfg.env_runner.env_config.headless = True

    pre_policy = _load_policy_from_ckpt(cfg.policy_pre, pre_train_cfg)
    post_policy = _load_policy_from_ckpt(cfg.policy_post, post_train_cfg)

    policy = TwoPhasePolicy(
        pre_policy,
        post_policy,
        gripper_thr=cfg.switch.get("gripper_thr", 0.5),
        closed_steps_to_switch=cfg.switch.get("closed_steps_to_switch", 3),
    )

    env_runner = RLBenchRunner(**cfg.env_runner)
    success_list, steps_list = env_runner.run(policy)

    n = len(success_list)
    n_success = sum(success_list)
    acc = n_success / n if n else 0.0
    print(f"Accuracy: {n_success}/{n} ({100.0 * acc:.1f}%)")
    if steps_list:
        avg_steps = sum(steps_list) / len(steps_list)
        print(f"Avg steps (successful): {avg_steps:.1f}")

    # Persist results to file (so running in a detached terminal is OK)
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pre_name = str(cfg.policy_pre.ckpt_name)
    post_name = str(cfg.policy_post.ckpt_name)
    base = f"validate_accuracy_two_phase_{pre_name}__{post_name}__{ts}"
    out_txt = results_dir / f"{base}.txt"
    out_json = results_dir / f"{base}.json"

    summary = {
        "timestamp": ts,
        "seed": int(cfg.seed),
        "num_episodes": int(cfg.env_runner.num_episodes),
        "max_episode_length": int(cfg.env_runner.max_episode_length),
        "pre": {
            "ckpt_name": pre_name,
            "ckpt_episode": str(cfg.policy_pre.get("ckpt_episode", "latest")),
            "num_k_infer": int(cfg.policy_pre.get("num_k_infer", 50)),
        },
        "post": {
            "ckpt_name": post_name,
            "ckpt_episode": str(cfg.policy_post.get("ckpt_episode", "latest")),
            "num_k_infer": int(cfg.policy_post.get("num_k_infer", 50)),
        },
        "switch": {
            "gripper_thr": float(cfg.switch.get("gripper_thr", 0.5)),
            "closed_steps_to_switch": int(cfg.switch.get("closed_steps_to_switch", 3)),
        },
        "success_count": int(n_success),
        "success_rate": float(acc),
        "success_rate_pct": float(100.0 * acc),
        "success_list": [bool(x) for x in success_list],
        "steps_list_successful": [int(x) for x in steps_list],
    }

    out_txt.write_text(
        "\n".join(
            [
                f"pre:  {pre_name} ({summary['pre']['ckpt_episode']})",
                f"post: {post_name} ({summary['post']['ckpt_episode']})",
                f"episodes: {n}",
                f"success: {n_success}/{n} ({100.0 * acc:.1f}%)",
                f"switch: thr={summary['switch']['gripper_thr']}  closed_steps_to_switch={summary['switch']['closed_steps_to_switch']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[results] wrote {out_txt}")
    print(f"[results] wrote {out_json}")

    return success_list


if __name__ == "__main__":
    main()

