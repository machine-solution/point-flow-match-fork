import torch
import numpy as np
import time
from collections import deque
from abc import ABC, abstractmethod
from pfp import DEVICE
from pfp.common.inference_profiling import (
    blank_action_profile,
    blank_infer_y_profile,
    cuda_sync,
    merge_action_profile,
)


class BasePolicy(ABC):
    """
    The base abstract class for all policies.
    """

    def __init__(self, n_obs_steps: int, subs_factor: int = 1) -> None:
        maxlen = n_obs_steps * subs_factor - (subs_factor - 1)
        self.obs_list = deque(maxlen=maxlen)
        self.robot_state_list = deque(maxlen=maxlen)
        self.subs_factor = subs_factor
        self.profile_inference = False
        self._last_action_profile: dict[str, float] = blank_action_profile()
        self._last_infer_y_profile: dict[str, float] = blank_infer_y_profile()
        return

    def enable_profile_inference(self, enabled: bool = True) -> None:
        self.profile_inference = bool(enabled)

    def reset_action_profile(self) -> None:
        self._last_action_profile = blank_action_profile()
        self._last_infer_y_profile = blank_infer_y_profile()

    def get_last_action_profile(self) -> dict[str, float]:
        return merge_action_profile(self._last_action_profile, self._last_infer_y_profile)

    def reset_obs(self):
        self.obs_list.clear()
        self.robot_state_list.clear()
        return

    def update_obs_lists(self, obs: np.ndarray, robot_state: np.ndarray):
        self.obs_list.append(obs)
        if len(self.obs_list) < self.obs_list.maxlen:
            self.obs_list.extendleft(
                [self.obs_list[0]] * (self.obs_list.maxlen - len(self.obs_list))
            )
        self.robot_state_list.append(robot_state)
        if len(self.robot_state_list) < self.robot_state_list.maxlen:
            n = self.robot_state_list.maxlen - len(self.robot_state_list)
            self.robot_state_list.extendleft([self.robot_state_list[0]] * n)
        return

    def sample_stacked_obs(self) -> tuple[np.ndarray, ...]:
        obs_stacked = np.stack(self.obs_list, axis=0)[:: self.subs_factor]
        robot_state_stacked = np.stack(self.robot_state_list, axis=0)[:: self.subs_factor]
        return obs_stacked, robot_state_stacked

    def predict_action(self, obs: np.ndarray, robot_state: np.ndarray) -> np.ndarray:
        prof = blank_action_profile() if self.profile_inference else None
        cuda_sync()
        t_total0 = time.perf_counter()

        if prof is not None:
            cuda_sync()
            t0 = time.perf_counter()
            self.update_obs_lists(obs, robot_state)
            cuda_sync()
            prof["update_obs_lists_ms"] = (time.perf_counter() - t0) * 1000.0

            cuda_sync()
            t0 = time.perf_counter()
            obs_stacked, robot_state_stacked = self.sample_stacked_obs()
            cuda_sync()
            prof["sample_stacked_obs_ms"] = (time.perf_counter() - t0) * 1000.0
        else:
            self.update_obs_lists(obs, robot_state)
            obs_stacked, robot_state_stacked = self.sample_stacked_obs()

        action = self.infer_from_np(obs_stacked, robot_state_stacked, profile=prof)

        if prof is not None:
            cuda_sync()
            prof["total_predict_action_ms"] = (time.perf_counter() - t_total0) * 1000.0
            self._last_action_profile = prof
        return action

    def infer_from_np(
        self,
        obs: np.ndarray,
        robot_state: np.ndarray,
        profile: dict[str, float] | None = None,
    ) -> np.ndarray:
        active = profile if profile is not None else (blank_action_profile() if self.profile_inference else None)

        if active is not None:
            cuda_sync()
            t0 = time.perf_counter()
            obs_th = torch.tensor(obs, device=DEVICE).unsqueeze(0)
            robot_state_th = torch.tensor(robot_state, device=DEVICE).unsqueeze(0)
            cuda_sync()
            active["numpy_to_torch_ms"] += (time.perf_counter() - t0) * 1000.0

            cuda_sync()
            t0 = time.perf_counter()
            obs_th = self._norm_obs(obs_th)
            robot_state_th = self._norm_robot_state(robot_state_th)
            cuda_sync()
            active["normalization_ms"] += (time.perf_counter() - t0) * 1000.0
        else:
            obs_th = torch.tensor(obs, device=DEVICE).unsqueeze(0)
            robot_state_th = torch.tensor(robot_state, device=DEVICE).unsqueeze(0)
            obs_th = self._norm_obs(obs_th)
            robot_state_th = self._norm_robot_state(robot_state_th)

        if active is not None:
            cuda_sync()
            t0 = time.perf_counter()
            ny = self.infer_y(obs_th, robot_state_th, return_traj=True)
            cuda_sync()
            active["infer_y_ms"] += (time.perf_counter() - t0) * 1000.0
        else:
            ny = self.infer_y(obs_th, robot_state_th, return_traj=True)

        if active is not None:
            cuda_sync()
            t0 = time.perf_counter()
            ny = self._denorm_robot_state(ny)
            ny = ny.squeeze().detach().cpu().numpy()
            cuda_sync()
            active["denormalization_ms"] += (time.perf_counter() - t0) * 1000.0
            if profile is None:
                self._last_action_profile = active
        else:
            ny = self._denorm_robot_state(ny)
            ny = ny.squeeze().detach().cpu().numpy()
        return ny

    @abstractmethod
    def _norm_obs(self, obs: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def _norm_robot_state(self, robot_state: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def _denorm_robot_state(self, robot_state: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def infer_y(
        self, obs: torch.Tensor, robot_state: torch.Tensor, return_traj: bool
    ) -> torch.Tensor:
        pass
