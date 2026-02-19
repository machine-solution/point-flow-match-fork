import os
import sys

# Add diffusion_policy to path if not already there (same as evaluate.py, record_actions.py)
_diffusion_policy_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "diffusion_policy")
if os.path.exists(_diffusion_policy_path) and _diffusion_policy_path not in sys.path:
    sys.path.insert(0, _diffusion_policy_path)

import hydra
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from rlbench.backend.observation import Observation
from rlbench.backend import waypoints as _rlbench_waypoints

# When planner returns -1 PyRep raises RuntimeError; RLBench only catches ConfigurationPathError.
# Treat -1 as infeasible so built-in retry (40 placements × 10 demo attempts) runs.
_orig_point_get_path = _rlbench_waypoints.Point.get_path
def _point_get_path_infeasible_on_neg1(self, ignore_collisions=False):
    try:
        return _orig_point_get_path(self, ignore_collisions)
    except RuntimeError as e:
        if "Return value: -1" in str(e) or "V-REP" in str(e):
            return None
        raise
_rlbench_waypoints.Point.get_path = _point_get_path_infeasible_on_neg1

from pfp import DATA_DIRS, set_seeds
from pfp.envs.rlbench_env import RLBenchEnv
from pfp.data.replay_buffer import RobotReplayBuffer
from pfp.common.visualization import RerunViewer as RV
from rlbench.backend.exceptions import TaskEnvironmentError


# For valid, call it with: --config-name=collect_demos_valid
# To actually save the data, remember to call it with: save_data=True
@hydra.main(version_base=None, config_path="../conf", config_name="collect_demos_train")
def main(cfg: OmegaConf):
    set_seeds(cfg.seed)
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    print(OmegaConf.to_yaml(cfg))

    assert cfg.mode in ["train", "valid"]
    if cfg.env_config.vis:
        RV("pfp_collect_demos")
    env = RLBenchEnv(use_pc_color=True, **cfg.env_config)
    if cfg.save_data:
        data_path = DATA_DIRS.PFP / cfg.env_config.task_name / cfg.mode
        if data_path.is_dir():
            print(f"ERROR: Data path {data_path} already exists! Exiting...")
            return
        replay_buffer = RobotReplayBuffer.create_from_path(data_path, mode="a")

    for _ in tqdm(range(cfg.num_episodes)):
        data_history = list()
        # RLBench's max_attempts only retries on get_demo() failure; reset() failure is outside that try.
        # So retry here when placement is infeasible (TaskEnvironmentError).
        for _ in range(500):
            try:
                demo = env.task.get_demos(1, live_demos=True, max_attempts=10)[0]
                break
            except TaskEnvironmentError:
                continue
        else:
            raise RuntimeError("Could not get a feasible demo after 500 attempts.")
        observations: list[Observation] = demo._observations
        for obs in observations:
            robot_state = env.get_robot_state(obs)
            images = env.get_images(obs)
            pcd = env.get_pcd(obs)
            pcd_xyz = np.asarray(pcd.points)
            pcd_color = np.asarray(pcd.colors)
            data_history.append(
                {
                    "pcd_xyz": pcd_xyz.astype(np.float32),
                    "pcd_color": (pcd_color * 255).astype(np.uint8),
                    "robot_state": robot_state.astype(np.float32),
                    "images": images,
                }
            )
            env.vis_step(robot_state, np.concatenate((pcd_xyz, pcd_color), axis=-1))

        if cfg.save_data:
            replay_buffer.add_episode_from_list(data_history, compressors="disk")
            print(f"Saved episode with {len(data_history)} steps to disk.")

        # while True:
        #     env.step(robot_state)
    return


if __name__ == "__main__":
    main()
