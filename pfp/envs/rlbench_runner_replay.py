import json
from pathlib import Path

import numpy as np
import wandb
from tqdm import tqdm

from pfp import set_seeds
from pfp.envs.rlbench_env import RLBenchEnv


class RLBenchRunnerReplay:
    """
    Replay runner that executes *recorded* actions and measures success in headless mode.

    This decouples:
      1) policy inference / action generation (can be slow)
      2) evaluation / visualization (can be repeated from the same saved actions)
    """

    def __init__(
        self,
        input_file: str,
        env_config: dict,
        verbose: bool = False,
    ) -> None:
        env_cfg = env_config.copy()
        env_cfg["headless"] = True
        env_cfg["vis"] = False
        self.env: RLBenchEnv = RLBenchEnv(**env_cfg)
        self.verbose = verbose
        self.input_file = Path(input_file)
        if not self.input_file.exists():
            raise FileNotFoundError(f"Recorded actions file not found: {self.input_file}")
        with self.input_file.open("r") as f:
            self.recorded_data = json.load(f)
        self.episodes = self.recorded_data.get("episodes", [])

    def run(self):
        wandb.define_metric("success", summary="mean")
        wandb.define_metric("steps", summary="mean")
        success_list: list[bool] = []
        steps_list: list[int] = []

        for ep in tqdm(self.episodes, desc="Replay"):
            episode_seed = ep.get("episode_seed", None)
            if episode_seed is not None:
                set_seeds(int(episode_seed))
            self.env.reset()

            actions = ep.get("actions", [])
            success = False
            steps = 0
            for i, a in enumerate(actions):
                next_robot_state = np.array(a, dtype=np.float32)
                reward, terminate = self.env.step(next_robot_state)
                success = bool(reward)
                steps = i + 1
                if success or terminate:
                    break

            if steps == 0:
                steps = len(actions)

            success_list.append(success)
            if success:
                steps_list.append(steps)
            if self.verbose:
                print(
                    f"Episode {ep.get('episode', '?')}: steps={steps} success={success} "
                    f"(recorded_success={ep.get('success', None)})"
                )
            wandb.log({"episode": ep.get("episode", 0), "success": int(success), "steps": steps})

        return success_list, steps_list

