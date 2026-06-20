import time
import wandb
import statistics
from tqdm import tqdm
from pfp.envs.rlbench_env import RLBenchEnv
from pfp.policy.base_policy import BasePolicy
from pfp.common.inference_profiling import cuda_sync

RESET_RETRIES = 20
RESET_RETRY_DELAY = 1.0

PROFILE_CSV_COLUMNS = [
    "method",
    "K",
    "episode_id",
    "step_id",
    "success_so_far",
    "predict_action_ms",
    "update_obs_lists_ms",
    "sample_stacked_obs_ms",
    "numpy_to_torch_ms",
    "normalization_ms",
    "encode_obs_ms",
    "infer_y_ms",
    "denormalization_ms",
    "nfe",
    "unet_total_ms",
    "unet_ms_per_nfe",
    "scheduler_ms",
    "extra_mlp_ms",
    "clone_detach_ms",
    "other_loop_ms",
    "env_step_ms",
    "total_step_ms",
]


class RLBenchRunner:
    def __init__(
        self,
        num_episodes: int,
        max_episode_length: int,
        env_config: dict,
        verbose=False,
        profile_inference_timing: bool = False,
        profile_method: str = "",
        profile_k: int = 0,
    ) -> None:
        self.env: RLBenchEnv = RLBenchEnv(**env_config)
        self.num_episodes = num_episodes
        self.max_episode_length = max_episode_length
        self.verbose = verbose
        self.profile_inference_timing = profile_inference_timing
        self.profile_method = profile_method
        self.profile_k = int(profile_k)
        self.profile_rows: list[dict] = []
        return

    def run(self, policy: BasePolicy, return_diagnostics: bool = False):
        wandb.define_metric("success", summary="mean")
        wandb.define_metric("steps", summary="mean")
        success_list: list[bool] = []
        steps_list: list[int] = []
        steps_per_episode: list[int] = []
        inference_ms_per_action: list[float] = []
        nfe_per_action: list[float] = []
        episode_time_s: list[float] = []
        if hasattr(policy, "reset_inference_diagnostics"):
            policy.reset_inference_diagnostics()
        if self.profile_inference_timing and hasattr(policy, "enable_profile_inference"):
            policy.enable_profile_inference(True)
        self.env.reset_rng()
        try:
            for episode in tqdm(range(self.num_episodes)):
                ep_t0 = time.perf_counter()
                policy.reset_obs()
                if self.profile_inference_timing and hasattr(policy, "reset_action_profile"):
                    policy.reset_action_profile()
                reset_ok = False
                for _ in range(RESET_RETRIES):
                    try:
                        self.env.reset()
                        reset_ok = True
                        break
                    except RuntimeError as e:
                        if "Return value: -1" in str(e) or "call failed" in str(e).lower():
                            time.sleep(RESET_RETRY_DELAY)
                            continue
                        raise
                if not reset_ok:
                    success_list.append(False)
                    steps_per_episode.append(0)
                    if self.verbose:
                        print(f"Episode {episode}: reset failed after {RESET_RETRIES} retries, skipping")
                    wandb.log({"episode": episode, "success": 0, "steps": 0})
                    continue
                success = False
                for step in range(self.max_episode_length):
                    robot_state, obs = self.env.get_obs()
                    if self.profile_inference_timing:
                        cuda_sync()
                        t_pred0 = time.perf_counter()
                        prediction = policy.predict_action(obs, robot_state)
                        cuda_sync()
                        t_pred1 = time.perf_counter()
                        predict_action_ms = (t_pred1 - t_pred0) * 1000.0
                        inference_ms_per_action.append(predict_action_ms)

                        cuda_sync()
                        t_env0 = time.perf_counter()
                        self.env.vis_step(robot_state, obs, prediction)
                        next_robot_state = prediction[-1, 0]
                        reward, terminate = self.env.step(next_robot_state)
                        cuda_sync()
                        t_env1 = time.perf_counter()
                        env_step_ms = (t_env1 - t_env0) * 1000.0
                        total_step_ms = predict_action_ms + env_step_ms

                        prof = (
                            policy.get_last_action_profile()
                            if hasattr(policy, "get_last_action_profile")
                            else {}
                        )
                        row = {
                            "method": self.profile_method,
                            "K": self.profile_k,
                            "episode_id": int(episode),
                            "step_id": int(step),
                            "success_so_far": int(success),
                            "predict_action_ms": float(predict_action_ms),
                            "env_step_ms": float(env_step_ms),
                            "total_step_ms": float(total_step_ms),
                        }
                        for col in PROFILE_CSV_COLUMNS:
                            if col in row:
                                continue
                            row[col] = float(prof.get(col, 0.0))
                        if hasattr(policy, "last_infer_nfe"):
                            row["nfe"] = float(getattr(policy, "last_infer_nfe"))
                            nfe_per_action.append(row["nfe"])
                        self.profile_rows.append(row)
                    else:
                        infer_t0 = time.perf_counter()
                        prediction = policy.predict_action(obs, robot_state)
                        inference_ms_per_action.append((time.perf_counter() - infer_t0) * 1000.0)
                        if hasattr(policy, "last_infer_nfe"):
                            nfe_per_action.append(float(getattr(policy, "last_infer_nfe")))
                        self.env.vis_step(robot_state, obs, prediction)
                        next_robot_state = prediction[-1, 0]
                        reward, terminate = self.env.step(next_robot_state)

                    success = bool(reward)
                    if success or terminate:
                        break
                episode_time_s.append(time.perf_counter() - ep_t0)
                success_list.append(success)
                steps_per_episode.append(int(step))
                if success:
                    steps_list.append(step)
                if self.verbose:
                    print(f"Steps: {step}")
                    print(f"Success: {success}")
                wandb.log({"episode": episode, "success": int(success), "steps": step})
        finally:
            try:
                self.env.close()
            except Exception:
                pass
        if not return_diagnostics:
            if self.profile_inference_timing:
                return success_list, steps_list, steps_per_episode, self.profile_rows
            return success_list, steps_list, steps_per_episode
        policy_diag = (
            policy.get_inference_diagnostics()
            if hasattr(policy, "get_inference_diagnostics")
            else {}
        )
        diagnostics = {
            "mean_inference_ms": float(statistics.fmean(inference_ms_per_action))
            if inference_ms_per_action
            else 0.0,
            "std_inference_ms": float(statistics.pstdev(inference_ms_per_action))
            if len(inference_ms_per_action) > 1
            else 0.0,
            "nfe_per_action": float(statistics.fmean(nfe_per_action)) if nfe_per_action else 0.0,
            "mean_episode_time_s": float(statistics.fmean(episode_time_s)) if episode_time_s else 0.0,
            "num_actions": int(len(inference_ms_per_action)),
            "episode_time_s": episode_time_s,
            "policy_inference_diagnostics": policy_diag,
        }
        if self.profile_inference_timing:
            diagnostics["profile_rows"] = self.profile_rows
        return success_list, steps_list, steps_per_episode, diagnostics
