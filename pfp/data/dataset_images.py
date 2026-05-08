from __future__ import annotations
import torch
import numpy as np
from diffusion_policy.common.sampler import SequenceSampler
from pfp.data.replay_buffer import RobotReplayBuffer
from pfp import DATA_DIRS
from pfp.common.phase_utils import compute_phase_labels_np, phase_cfg_from, PhaseConditioningConfig


class RobotDatasetImages(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path: str,
        n_obs_steps: int,
        n_pred_steps: int,
        subs_factor: int = 1,  # 1 means no subsampling
        phase_conditioning: dict | None = None,
        **kwargs,
    ) -> None:
        """
        To me it makes sense that sequence_length == n_obs_steps + n_prediction_steps
        """
        replay_buffer = RobotReplayBuffer.create_from_path(data_path, mode="r")
        data_keys = ["robot_state", "images"]
        data_key_first_k = {"images": n_obs_steps * subs_factor}
        self.sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=(n_obs_steps + n_pred_steps) * subs_factor - (subs_factor - 1),
            pad_before=(n_obs_steps - 1) * subs_factor,
            pad_after=(n_pred_steps - 1) * subs_factor + (subs_factor - 1),
            keys=data_keys,
            key_first_k=data_key_first_k,
        )
        self.n_obs_steps = n_obs_steps
        self.n_prediction_steps = n_pred_steps
        self.subs_factor = subs_factor
        self.rng = np.random.default_rng()
        self.phase_cfg: PhaseConditioningConfig = phase_cfg_from(phase_conditioning)
        self._phase_stats = None
        return

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        sample: dict[str, np.ndarray] = self.sampler.sample_sequence(idx)
        cur_step_i = self.n_obs_steps * self.subs_factor
        images = sample["images"][: cur_step_i : self.subs_factor]
        robot_state_obs = sample["robot_state"][: cur_step_i : self.subs_factor]
        robot_state_pred = sample["robot_state"][cur_step_i :: self.subs_factor]
        if not self.phase_cfg.enabled:
            return images, robot_state_obs, robot_state_pred
        full_rs = sample["robot_state"][:: self.subs_factor].astype(np.float32)
        phase_full, _ = compute_phase_labels_np(
            full_rs,
            thr=self.phase_cfg.gripper_close_threshold,
            contact_window=self.phase_cfg.contact_window,
            num_phases=self.phase_cfg.num_phases,
        )
        phase_pred = phase_full[self.n_obs_steps : self.n_obs_steps + self.n_prediction_steps].astype(np.int64)
        if phase_pred.shape[0] != robot_state_pred.shape[0]:
            L = min(int(phase_pred.shape[0]), int(robot_state_pred.shape[0]))
            phase_pred = phase_pred[:L]
            robot_state_pred = robot_state_pred[:L]
        return images, robot_state_obs, robot_state_pred, phase_pred

    def phase_stats(self, *, max_episodes: int = 2000) -> dict | None:
        if not self.phase_cfg.enabled:
            return None
        if self._phase_stats is not None:
            return self._phase_stats
        rb = getattr(self.sampler, "replay_buffer", None)
        if rb is None:
            return None
        n_ep = int(getattr(rb, "n_episodes", 0))
        n_scan = min(n_ep, int(max_episodes))
        phase_counts = np.zeros((self.phase_cfg.num_phases,), dtype=np.int64)
        grasp_steps = []
        no_grasp = 0
        for ep_i in range(n_scan):
            ep = rb.get_episode(ep_i, copy=False)
            rs = np.asarray(ep["robot_state"], dtype=np.float32)
            ph, t = compute_phase_labels_np(
                rs,
                thr=self.phase_cfg.gripper_close_threshold,
                contact_window=self.phase_cfg.contact_window,
                num_phases=self.phase_cfg.num_phases,
            )
            for p in range(self.phase_cfg.num_phases):
                phase_counts[p] += int((ph == p).sum())
            if t is None:
                no_grasp += 1
            else:
                grasp_steps.append(int(t))
        grasp_steps_arr = np.asarray(grasp_steps, dtype=np.int64) if grasp_steps else None
        out = {
            "enabled": True,
            "num_phases": int(self.phase_cfg.num_phases),
            "contact_window": int(self.phase_cfg.contact_window),
            "gripper_close_threshold": float(self.phase_cfg.gripper_close_threshold),
            "episodes_scanned": int(n_scan),
            "episodes_total": int(n_ep),
            "no_grasp_episodes": int(no_grasp),
            "phase_counts": phase_counts.tolist(),
            "phase_fracs": (phase_counts / max(1, int(phase_counts.sum()))).tolist(),
            "avg_grasp_timestep": float(grasp_steps_arr.mean()) if grasp_steps_arr is not None else None,
            "median_grasp_timestep": float(np.median(grasp_steps_arr)) if grasp_steps_arr is not None else None,
        }
        self._phase_stats = out
        return out


if __name__ == "__main__":
    dataset = RobotDatasetImages(
        data_path=DATA_DIRS.PFP / "open_fridge" / "train",
        n_obs_steps=2,
        n_pred_steps=8,
        subs_factor=5,
    )
    i = 20
    obs, robot_state_obs, robot_state_pred = dataset[i]
    print("robot_state_obs: ", robot_state_obs)
    print("robot_state_pred: ", robot_state_pred)
    print("done")
