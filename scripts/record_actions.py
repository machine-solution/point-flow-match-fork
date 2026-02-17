import sys
import os
# Add diffusion_policy to path if not already there
diffusion_policy_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "diffusion_policy")
if os.path.exists(diffusion_policy_path) and diffusion_policy_path not in sys.path:
    sys.path.insert(0, diffusion_policy_path)

import hydra
import wandb
import subprocess
from omegaconf import OmegaConf, open_dict
from pfp import set_seeds, REPO_DIRS
from pfp.envs.rlbench_runner_record import RLBenchRunnerRecord
from pfp.policy.base_policy import BasePolicy


@hydra.main(version_base=None, config_path="../conf", config_name="eval")
def main(cfg: OmegaConf):
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg))
    set_seeds(cfg.seed)

    # Download checkpoint if not present
    ckpt_path = REPO_DIRS.CKPT / cfg.policy.ckpt_name
    if not ckpt_path.exists():
        subprocess.run(
            [
                "rsync",
                "-hPrl",
                f"chisari@rlgpu2:{ckpt_path}",
                f"{REPO_DIRS.CKPT}/",
            ]
        )

    with open_dict(cfg):
        train_cfg = OmegaConf.load(ckpt_path / "config.yaml")
        cfg.model = train_cfg.model
        cfg.env_runner.env_config.task_name = train_cfg.task_name
        cfg.env_runner.env_config.obs_mode = train_cfg.obs_mode
        cfg.env_runner.env_config.use_pc_color = train_cfg.dataset.use_pc_color
        cfg.env_runner.env_config.n_points = train_cfg.dataset.n_points

    print(OmegaConf.to_yaml(cfg))

    wandb.init(
        project="pfp-eval-rebuttal",
        entity="rl-lab-chisari",
        config=OmegaConf.to_container(cfg),
        mode="online" if cfg.log_wandb else "disabled",
    )
    
    # Load policy class and call load_from_checkpoint directly
    policy_class = hydra.utils.get_class(train_cfg.model._target_)
    policy: BasePolicy = policy_class.load_from_checkpoint(
        ckpt_name=cfg.policy.ckpt_name,
        ckpt_episode=cfg.policy.get("ckpt_episode", "latest"),
        num_k_infer=cfg.policy.get("num_k_infer", 50),
        flow_schedule=cfg.policy.get("flow_schedule", None),
        exp_scale=cfg.policy.get("exp_scale", None),
        subs_factor=cfg.policy.get("subs_factor", 1),
    )
    
    # Use recording runner (headless, fast)
    # Get output_file from config (may be set via +output_file=...)
    output_file = cfg.get("output_file", f"recorded_actions_{cfg.policy.ckpt_name}.json")
    # Если путь не содержит папку, пишем в recordings/
    if "/" not in str(output_file):
        output_file = os.path.join("recordings", output_file)
    with open_dict(cfg):
        if "output_file" not in cfg:
            cfg.output_file = output_file
    output_file = cfg.output_file

    # Pass base_seed so that per-episode seeds can be derived and recorded
    env_runner_config = dict(cfg.env_runner)
    env_runner_config["output_file"] = output_file
    env_runner_config["base_seed"] = int(cfg.seed)
    env_runner = RLBenchRunnerRecord(**env_runner_config)
    _ = env_runner.run(policy)
    wandb.finish()
    print(f"\nActions recorded to: {output_file}")
    print(f"To play back, run: python scripts/playback_actions.py input_file={output_file}")
    return


if __name__ == "__main__":
    main()
