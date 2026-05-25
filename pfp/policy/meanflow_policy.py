from __future__ import annotations

import logging
import time
import torch
import torch.nn as nn

from pfp import DEVICE
from pfp.common.phase_utils import compute_phase_labels_torch_from_gripper
from pfp.policy.fm_policy import FMPolicy

logger = logging.getLogger(__name__)


class MeanFlowPolicy(FMPolicy):
    """
    MeanFlow/one-step PointFlowMatch.

    Predicts interval-averaged velocity u_theta(x_r, r, t, obs), then one-step integrates:
        x_1 = x_0 + u_theta(x_0, r=0, t=1, obs)
    """

    def __init__(
        self,
        *args,
        interval_embed_dim: int = 64,
        interval_hidden_dim: int = 128,
        meanflow: dict | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.interval_embed_dim = int(interval_embed_dim)
        self.interval_hidden_dim = int(interval_hidden_dim)
        self.meanflow_enabled = True if meanflow is None else bool(meanflow.get("enabled", True))
        self.meanflow_one_step = True if meanflow is None else bool(meanflow.get("one_step", True))
        self.interval_mlp = nn.Sequential(
            nn.Linear(3, self.interval_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.interval_hidden_dim, self.interval_embed_dim),
        )
        # MeanFlow is designed as one-step. Keep API-compatible setter but pin NFE to 1.
        self.num_k_infer = 1

    def set_num_k_infer(self, num_k_infer: int):
        if int(num_k_infer) != 1:
            logger.warning("MeanFlowPolicy enforces one-step inference (requested num_k_infer=%s).", num_k_infer)
        self.num_k_infer = 1
        return

    def _interval_global_cond(self, nx: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B = nx.shape[0]
        interval_feat = torch.stack([r.view(B), t.view(B), (t - r).view(B)], dim=-1)
        interval_emb = self.interval_mlp(interval_feat)
        return torch.cat([nx, interval_emb], dim=-1)

    def _infer_phase_seq(
        self,
        nx: torch.Tensor,
        robot_state_obs: torch.Tensor,
        phase: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if not self.phase_enabled:
            return None
        B = nx.shape[0]
        if phase is not None:
            if phase.dtype != torch.int64:
                phase = phase.to(torch.int64)
            if phase.ndim == 1:
                return phase.view(B, 1).repeat(1, self.n_pred_steps)
            return phase
        if self.phase_pred_enabled:
            phase_logits = self.predict_phase_logits(nx)
            phase_current = phase_logits.argmax(dim=-1)
            return phase_current.view(B, 1).expand(B, self.n_pred_steps)
        if self._phase_rollout_cfg.enabled and self._phase_rollout_cfg.force_phase >= 0:
            p = int(min(max(self._phase_rollout_cfg.force_phase, 0), self.phase_cfg.num_phases - 1))
            return torch.full((B, self.n_pred_steps), p, device=DEVICE, dtype=torch.int64)
        if self._phase_rollout_cfg.enabled:
            p = int(min(max(self._rollout_phase_state, 0), self.phase_cfg.num_phases - 1))
            return torch.full((B, self.n_pred_steps), p, device=DEVICE, dtype=torch.int64)
        g_last_raw = self._denorm_gripper(robot_state_obs[:, -1, 9])
        ph, _ = compute_phase_labels_torch_from_gripper(
            g_last_raw.view(B, 1).repeat(1, self.n_pred_steps),
            thr=self.phase_cfg.gripper_close_threshold,
            contact_window=self.phase_cfg.contact_window,
            num_phases=self.phase_cfg.num_phases,
        )
        return ph

    def calculate_loss(
        self,
        pcd: torch.Tensor,
        robot_state_obs: torch.Tensor,
        robot_state_pred: torch.Tensor,
        *,
        phase: torch.Tensor | None = None,
    ):
        nx = self.encode_obs(pcd, robot_state_obs)
        ny: torch.Tensor = robot_state_pred
        B, T, _ = ny.shape
        phase_loss = None
        phase_metrics: dict[str, float] = {}
        phase_flow = None

        if self.phase_enabled:
            if phase is None:
                raise ValueError("phase_conditioning.enabled=true requires phase labels in calculate_loss().")
            phase = self._assert_valid_phase(phase, B, T)
            phase_flow = phase
            if self.phase_pred_enabled:
                phase_loss, phase_metrics = self._compute_phase_aux_loss(nx, phase)
                phase_flow = self._phase_for_flow_train(phase, nx, B=B, T=T)
            phase_metrics.update(self._phase_flow_train_metrics(phase_flow, phase))

        eps = 1e-3
        r = torch.rand((B, 1, 1), device=DEVICE) * (1.0 - eps)
        delta = torch.rand((B, 1, 1), device=DEVICE) * (1.0 - r)
        delta = torch.clamp(delta, min=eps)
        t = torch.clamp(r + delta, max=1.0)

        z0 = self._init_noise(B)
        z1 = ny
        x_r = (1.0 - r) * z0 + r * z1
        x_t = (1.0 - t) * z0 + t * z1
        u_target = (x_t - x_r) / torch.clamp(t - r, min=eps)

        D = int(self.y_dim)
        if self.phase_enabled:
            phase_emb = self.phase_embedding(phase_flow)
            x_in = torch.cat([x_r, phase_emb], dim=-1)
            assert x_in.shape[-1] == D + int(self.phase_cfg.phase_embed_dim)
        else:
            x_in = x_r

        timesteps = r.view(B) * self.pos_emb_scale if self.time_conditioning else None
        global_cond = self._interval_global_cond(nx, r, t)
        pred_u_full = self.diffusion_net(x_in, timesteps, global_cond=global_cond)
        pred_u = self._slice_velocity(pred_u_full)

        per_xyz, per_rot6d, per_grip = self._per_timestep_loss(pred_u, u_target)
        weights = self._compute_gripper_weights(ny)
        loss_xyz = (per_xyz * weights).mean()
        loss_rot6d = (per_rot6d * weights).mean()
        loss_grip = (per_grip * weights).mean()
        return loss_xyz, loss_rot6d, loss_grip, phase_loss, phase_metrics

    def infer_y(
        self,
        pcd: torch.Tensor,
        robot_state_obs: torch.Tensor,
        phase: torch.Tensor | None = None,
        noise=None,
        return_traj=False,
    ) -> torch.Tensor:
        t0 = time.perf_counter()

        nx = self.encode_obs(pcd, robot_state_obs)
        B = nx.shape[0]
        z0 = self._init_noise(B) if noise is None else noise
        phase_seq = self._infer_phase_seq(nx, robot_state_obs, phase)
        if self.phase_enabled:
            assert phase_seq is not None
            z_in = torch.cat([z0, self.phase_embedding(phase_seq)], dim=-1)
        else:
            z_in = z0

        r = torch.zeros((B, 1, 1), device=DEVICE)
        t = torch.ones((B, 1, 1), device=DEVICE)
        timesteps = r.view(B) * self.pos_emb_scale if self.time_conditioning else None
        global_cond = self._interval_global_cond(nx, r, t)
        u_pred_full = self.diffusion_net(z_in, timesteps, global_cond=global_cond)
        u_pred = self._slice_velocity(u_pred_full)
        pred = z0 + u_pred

        # Keep diagnostics API compatible with FMPolicy.
        self.last_infer_nfe = 1
        self._infer_calls_total += 1
        self._infer_actions_total += int(B)
        self._infer_nfe_total += int(B)
        self._infer_time_total_ms += float((time.perf_counter() - t0) * 1000.0)

        if return_traj:
            return torch.stack([z0, pred])
        return pred
