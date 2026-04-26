import sys
import os

# Add diffusion_policy to path if not already there (RLBench deps live there sometimes)
diffusion_policy_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "..", "diffusion_policy"
)
if os.path.exists(diffusion_policy_path) and diffusion_policy_path not in sys.path:
    sys.path.insert(0, diffusion_policy_path)

import json
from pathlib import Path

import hydra
import wandb
from omegaconf import OmegaConf, open_dict

from pfp import set_seeds
from pfp.envs.rlbench_runner_replay import RLBenchRunnerReplay


@hydra.main(version_base=None, config_path="../conf", config_name="eval")
def main(cfg: OmegaConf):
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    set_seeds(cfg.seed)
    wandb.init(mode="disabled")

    # input_file=... passed via CLI (same style as playback_actions.py)
    input_file = None
    with open_dict(cfg):
        if "input_file" in cfg:
            input_file = cfg.input_file
        else:
            for arg in sys.argv:
                if arg.startswith("input_file="):
                    input_file = arg.split("=", 1)[1]
                    cfg.input_file = input_file
                    break
    if not input_file:
        raise ValueError("input_file must be specified. Example: python scripts/replay_actions_accuracy.py input_file=recordings/foo.json")

    p = Path(input_file)
    if not p.exists():
        raise FileNotFoundError(f"Recorded actions file not found: {p}")
    recorded = json.loads(p.read_text())
    env_config = recorded.get("env_config", {})
    # Add required defaults
    env_config.setdefault("voxel_size", 0.01)
    env_config.setdefault("headless", True)
    env_config.setdefault("vis", False)

    env_runner = RLBenchRunnerReplay(input_file=str(p), env_config=env_config, verbose=cfg.get("verbose", False))
    success_list, steps_list = env_runner.run()

    n = len(success_list)
    n_success = sum(success_list)
    acc = n_success / n if n else 0.0
    print(f"Replay accuracy: {n_success}/{n} ({100.0 * acc:.1f}%)")
    if steps_list:
        print(f"Avg steps (successful): {sum(steps_list) / len(steps_list):.1f}")
    return


if __name__ == "__main__":
    main()

