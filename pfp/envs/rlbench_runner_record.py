import json
from pathlib import Path

import numpy as np
import wandb
from tqdm import tqdm

from pfp import set_seeds
from pfp.envs.rlbench_env import RLBenchEnv
from pfp.policy.base_policy import BasePolicy


class RLBenchRunnerRecord:
    """
    Runner that generates actions and records them to a file.
    Uses headless mode for faster execution (no GUI rendering).

    Additionally records per-episode seeds so that the exact
    randomization of each task instance can be reproduced later.
    """

    def __init__(
        self,
        num_episodes: int,
        max_episode_length: int,
        env_config: dict,
        verbose: bool = False,
        output_file: str = "recorded_actions.json",
        base_seed: int | None = None,
    ) -> None:
        # Force headless mode for recording (faster)
        env_config_record = env_config.copy()
        env_config_record["headless"] = True
        env_config_record["vis"] = False

        self.env: RLBenchEnv = RLBenchEnv(**env_config_record)
        self.num_episodes = num_episodes
        self.max_episode_length = max_episode_length
        self.verbose = verbose
        self.output_file = Path(output_file)
        # Seed from which per-episode seeds will be derived (seed + episode_idx)
        self.base_seed = base_seed
        self.recorded_episodes = []
        return

    def run(self, policy: BasePolicy):
        """
        Generate actions using policy in headless mode.
        Record all actions and per-episode metadata to file.
        """
        print(f"Recording actions to {self.output_file}")
        print("Using headless mode (fast, no GUI)")

        wandb.define_metric("success", summary="mean")
        wandb.define_metric("steps", summary="mean")
        success_list: list[bool] = []
        steps_list: list[int] = []
        self.env.reset_rng()

        # Common metadata for all episode files
        common_env_config = {
            "task_name": self.env.task.get_name(),
            "obs_mode": self.env.obs_mode,
            "n_points": self.env.n_points,
            "use_pc_color": self.env.use_pc_color,
        }

        # Directory to store per‑episode snapshots, e.g.
        # recordings/recorded_actions_.../ep_000.json, ep_001.json, ...
        episodes_dir = self.output_file.parent / self.output_file.stem
        episodes_dir.mkdir(parents=True, exist_ok=True)

        for episode in tqdm(range(self.num_episodes), desc="Recording"):
            # Derive deterministic per-episode seed if base_seed is provided
            episode_seed = None
            if self.base_seed is not None:
                episode_seed = int(self.base_seed) + int(episode)
                set_seeds(episode_seed)

            policy.reset_obs()
            self.env.reset()
            episode_data = {
                "episode": episode,
                "episode_seed": episode_seed,
                "actions": [],
                "robot_states": [],
                "success": False,
                "steps": 0,
            }

            for step in range(self.max_episode_length):
                robot_state, obs = self.env.get_obs()
                prediction = policy.predict_action(obs, robot_state)
                next_robot_state = prediction[-1, 0]  # Last K step, first T step

                # Record action and robot state
                episode_data["actions"].append(next_robot_state.tolist())
                episode_data["robot_states"].append(robot_state.tolist())

                reward, terminate = self.env.step(next_robot_state)
                success = bool(reward)
                if success or terminate:
                    episode_data["success"] = success
                    episode_data["steps"] = step + 1
                    break

            if episode_data["steps"] == 0:
                episode_data["steps"] = self.max_episode_length

            # Keep in-memory aggregate
            self.recorded_episodes.append(episode_data)
            success_list.append(episode_data["success"])
            if episode_data["success"]:
                steps_list.append(episode_data["steps"])
            if self.verbose:
                print(
                    f"Episode {episode}: Steps: {episode_data['steps']}, "
                    f"Success: {episode_data['success']}, Seed: {episode_seed}"
                )
            wandb.log(
                {
                    "episode": episode,
                    "success": int(episode_data["success"]),
                    "steps": episode_data["steps"],
                }
            )

            # ---- Save this episode separately so a crash later doesn't lose it ----
            single_output = {
                "num_episodes": 1,
                "max_episode_length": self.max_episode_length,
                "base_seed": self.base_seed,
                "env_config": common_env_config,
                "episodes": [episode_data],
            }
            ep_path = episodes_dir / f"ep_{episode:03d}.json"
            with ep_path.open("w") as f_ep:
                json.dump(single_output, f_ep, indent=2)

        # Ensure parent directory exists (e.g. recordings/)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        # Save to file
        output_data = {
            "num_episodes": self.num_episodes,
            "max_episode_length": self.max_episode_length,
            "base_seed": self.base_seed,
            "env_config": {
                "task_name": self.env.task.get_name(),
                "obs_mode": self.env.obs_mode,
                "n_points": self.env.n_points,
                "use_pc_color": self.env.use_pc_color,
            },
            "episodes": self.recorded_episodes,
        }

        with open(self.output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"\nRecorded {self.num_episodes} episodes to {self.output_file}")
        print(f"Success rate: {sum(success_list)}/{len(success_list)}")
        return success_list, steps_list
