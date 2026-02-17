import wandb
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from pfp.envs.rlbench_env import RLBenchEnv
from pfp.policy.base_policy import BasePolicy


class RLBenchRunnerPlayback:
    """
    Runner that plays back recorded actions from a file.
    Launches CoppeliaSim with GUI for visualization.
    This is fast because actions are pre-generated - no policy inference needed.
    """
    def __init__(
        self,
        input_file: str,
        env_config: dict,
        verbose=False,
    ) -> None:
        # Force visualization mode for playback
        env_config_playback = env_config.copy()
        env_config_playback["vis"] = True
        env_config_playback["headless"] = False
        
        self.env: RLBenchEnv = RLBenchEnv(**env_config_playback)
        # Give CoppeliaSim extra time to fully initialize in GUI mode
        import time
        time.sleep(2.0)
        self.verbose = verbose
        self.input_file = Path(input_file)
        
        # Load recorded actions
        if not self.input_file.exists():
            raise FileNotFoundError(f"Recorded actions file not found: {self.input_file}")
        
        with open(self.input_file, 'r') as f:
            self.recorded_data = json.load(f)
        
        self.num_episodes = self.recorded_data["num_episodes"]
        self.recorded_episodes = self.recorded_data["episodes"]
        
        print(f"Loaded {self.num_episodes} episodes from {self.input_file}")
        print("Playing back with visualization (GUI mode)")
        return

    def run(self, policy: BasePolicy = None):
        """
        Play back recorded actions in CoppeliaSim with visualization.
        Policy is not used - we just execute pre-recorded actions.
        """
        wandb.define_metric("success", summary="mean")
        wandb.define_metric("steps", summary="mean")
        success_list: list[bool] = []
        steps_list: list[int] = []
        
        import time
        
        for episode_idx, episode_data in enumerate(tqdm(self.recorded_episodes, desc="Playback")):
            self.env.reset()
            # Give CoppeliaSim time to fully initialize after reset
            time.sleep(1.0)
            
            actions = episode_data["actions"]
            recorded_success = episode_data["success"]
            recorded_steps = episode_data["steps"]
            
            for step, action in enumerate(actions):
                try:
                    # Convert action from list to numpy array
                    next_robot_state = np.array(action, dtype=np.float32)
                    
                    # Get observation first (same order as original runner)
                    robot_state, obs = self.env.get_obs()
                    
                    # Visualize (if visualization is enabled)
                    if self.env.vis:
                        try:
                            # Create dummy prediction for vis_step
                            dummy_prediction = np.zeros((1, 1, 10), dtype=np.float32)  # (K=1, T=1, 10)
                            dummy_prediction[0, 0] = next_robot_state
                            self.env.vis_step(robot_state, obs, dummy_prediction)
                        except Exception as vis_e:
                            # If visualization fails, continue anyway
                            if self.verbose:
                                print(f"Warning: Visualization error at step {step}: {vis_e}")
                    
                    # Execute recorded action (same order as original runner)
                    reward, terminate = self.env.step(next_robot_state)
                    success = bool(reward)
                    
                    # Small delay for visualization to keep up
                    if self.env.vis:
                        time.sleep(0.01)  # 10ms delay for smooth visualization
                    
                    if success or terminate:
                        break
                except RuntimeError as e:
                    if "V-REP" in str(e) or "call failed" in str(e).lower():
                        print(f"Warning: CoppeliaSim error at step {step}: {e}")
                        print("Skipping this episode due to simulator error")
                        success = False
                        break
                    else:
                        raise
                except Exception as e:
                    print(f"Error at step {step}: {e}")
                    print("Skipping this episode")
                    success = False
                    break
            
            success_list.append(success)
            if success:
                steps_list.append(step + 1)
            if self.verbose:
                print(f"Episode {episode_idx}: Steps: {step + 1}, Success: {success}")
                print(f"  (Recorded: Steps: {recorded_steps}, Success: {recorded_success})")
            wandb.log({"episode": episode_idx, "success": int(success), "steps": step + 1})
        
        print(f"\nPlayback complete: {sum(success_list)}/{len(success_list)} successful")
        return success_list, steps_list
