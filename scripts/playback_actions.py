import sys
import os
# Add diffusion_policy to path if not already there
diffusion_policy_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "diffusion_policy")
if os.path.exists(diffusion_policy_path) and diffusion_policy_path not in sys.path:
    sys.path.insert(0, diffusion_policy_path)

import hydra
import wandb
from omegaconf import OmegaConf, open_dict
from pfp import set_seeds, REPO_DIRS
from pfp.envs.rlbench_runner_playback import RLBenchRunnerPlayback
from pfp.common.visualization import RerunViewer as RV


@hydra.main(version_base=None, config_path="../conf", config_name="eval")
def main(cfg: OmegaConf):
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    
    # Get input file from config (may be set via +input_file=...)
    with open_dict(cfg):
        if "input_file" not in cfg:
            # Try to get from command line args or raise error
            import sys
            for arg in sys.argv:
                if arg.startswith("input_file="):
                    input_file = arg.split("=", 1)[1]
                    cfg.input_file = input_file
                    break
            else:
                raise ValueError("input_file must be specified. Example: python scripts/playback_actions.py +input_file=recorded_actions.json")
        input_file = cfg.input_file
    
    print(f"Playing back actions from: {input_file}")
    
    # Load recorded data to get env config
    import json
    with open(input_file, 'r') as f:
        recorded_data = json.load(f)
    
    # Set up env config from recorded data
    with open_dict(cfg):
        if "env_runner" not in cfg:
            cfg.env_runner = OmegaConf.create({})
        if "env_config" not in cfg.env_runner:
            cfg.env_runner.env_config = OmegaConf.create({})
        
        # Use config from recorded file
        env_config_recorded = recorded_data.get("env_config", {})
        cfg.env_runner.env_config.task_name = env_config_recorded.get("task_name", "open_fridge")
        cfg.env_runner.env_config.obs_mode = env_config_recorded.get("obs_mode", "pcd")
        cfg.env_runner.env_config.n_points = env_config_recorded.get("n_points", 4096)
        cfg.env_runner.env_config.use_pc_color = env_config_recorded.get("use_pc_color", False)
        cfg.env_runner.env_config.voxel_size = 0.01
        cfg.env_runner.env_config.vis = True
        cfg.env_runner.env_config.headless = False
        cfg.env_runner.num_episodes = recorded_data.get("num_episodes", 1)
        cfg.env_runner.max_episode_length = recorded_data.get("max_episode_length", 200)
        cfg.env_runner.verbose = True
    
    print(OmegaConf.to_yaml(cfg))

    if cfg.env_runner.env_config.vis:
        RV("pfp_playback")
    wandb.init(
        project="pfp-eval-rebuttal",
        entity="rl-lab-chisari",
        config=OmegaConf.to_container(cfg),
        mode="disabled",  # Disable wandb for playback
    )
    
    # Use playback runner (with GUI)
    env_runner = RLBenchRunnerPlayback(
        input_file=input_file,
        env_config=dict(cfg.env_runner.env_config),
        verbose=cfg.env_runner.get("verbose", False),
    )
    _ = env_runner.run()
    wandb.finish()
    return


if __name__ == "__main__":
    main()
