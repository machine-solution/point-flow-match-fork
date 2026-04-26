import sys
import os

# Add diffusion_policy to path if not already there
diffusion_policy_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "..", "diffusion_policy"
)
if os.path.exists(diffusion_policy_path) and diffusion_policy_path not in sys.path:
    sys.path.insert(0, diffusion_policy_path)

import hydra
import wandb
from omegaconf import OmegaConf, open_dict
from pfp import set_seeds, REPO_DIRS
from pfp.envs.rlbench_runner_record import RLBenchRunnerRecord
from pfp.policy.base_policy import BasePolicy
from pfp.policy.two_phase_policy import TwoPhasePolicy


def _load_policy(ckpt_name: str, ckpt_episode: str, num_k_infer: int, train_cfg) -> BasePolicy:
    policy_class = hydra.utils.get_class(train_cfg.model._target_)
    return policy_class.load_from_checkpoint(
        ckpt_name=ckpt_name,
        ckpt_episode=ckpt_episode,
        num_k_infer=num_k_infer,
    )


@hydra.main(version_base=None, config_path="../conf", config_name="eval_two_phase")
def main(cfg: OmegaConf):
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg))
    set_seeds(cfg.seed)

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

        # Env config from pre
        cfg.env_runner.env_config.task_name = pre_train_cfg.task_name
        cfg.env_runner.env_config.obs_mode = pre_train_cfg.obs_mode
        cfg.env_runner.env_config.use_pc_color = pre_train_cfg.dataset.use_pc_color
        cfg.env_runner.env_config.n_points = pre_train_cfg.dataset.n_points

    wandb.init(mode="disabled" if not cfg.log_wandb else "online")

    pre_policy = _load_policy(
        cfg.policy_pre.ckpt_name,
        cfg.policy_pre.get("ckpt_episode", "latest"),
        int(cfg.policy_pre.get("num_k_infer", 50)),
        pre_train_cfg,
    )
    post_policy = _load_policy(
        cfg.policy_post.ckpt_name,
        cfg.policy_post.get("ckpt_episode", "latest"),
        int(cfg.policy_post.get("num_k_infer", 50)),
        post_train_cfg,
    )
    policy = TwoPhasePolicy(
        pre_policy,
        post_policy,
        gripper_thr=cfg.switch.get("gripper_thr", 0.5),
        closed_steps_to_switch=cfg.switch.get("closed_steps_to_switch", 3),
    )

    # Output file
    output_file = cfg.get(
        "output_file",
        f"recorded_actions_two_phase_{cfg.policy_pre.ckpt_name}__{cfg.policy_post.ckpt_name}.json",
    )
    if "/" not in str(output_file):
        output_file = os.path.join("recordings", output_file)
    with open_dict(cfg):
        cfg.output_file = output_file

    env_runner_config = dict(cfg.env_runner)
    env_runner_config["output_file"] = output_file
    env_runner_config["base_seed"] = int(cfg.seed)
    env_runner = RLBenchRunnerRecord(**env_runner_config)
    _ = env_runner.run(policy)

    wandb.finish()
    print(f"\nActions recorded to: {output_file}")
    print(f"To replay (headless metric), run: python scripts/replay_actions_accuracy.py input_file={output_file}")
    print(f"To visualize (GUI), run: python scripts/playback_actions.py input_file={output_file}")


if __name__ == "__main__":
    main()

