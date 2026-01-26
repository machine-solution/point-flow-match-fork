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
from pfp.envs.rlbench_runner import RLBenchRunner
from pfp.policy.base_policy import BasePolicy
from pfp.common.visualization import RerunViewer as RV


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

    if cfg.env_runner.env_config.vis:
        RV("pfp_evaluate")
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
    env_runner = RLBenchRunner(**cfg.env_runner)
    _ = env_runner.run(policy)
    wandb.finish()
    return


if __name__ == "__main__":
    main()
