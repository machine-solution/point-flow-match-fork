from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import torch
import torch.nn as nn

from pfp import DEVICE
from pfp.common.inference_profiling import blank_infer_y_profile, cuda_sync
from pfp.common.momentum_meanflow_utils import (
    MOMENTUM_INFER_SCHEDULES,
    build_inference_time_grid,
    sample_three_times,
    sample_two_times,
)
from pfp.policy.fm_policy import FMPolicy

logger = logging.getLogger(__name__)


@dataclass
class MomentumMeanFlowConfig:
    interval_embed_dim: int = 64
    interval_hidden_dim: int = 128
    lambda_correct: float = 1.0
    dt_min: float = 1.0 / 32.0
    momentum_meanflow_schedule: str = "uniform"


class MomentumMeanFlowPolicy(FMPolicy):
    """
    Self-correcting / momentum-conditioned MeanFlow.

    Trains:
      - first-step interval velocity on arbitrary [t0, t1]
      - corrective second-step velocity from model-predicted x_hat_{t1}
    Inference:
      multi-step integration with previous predicted velocity prev_u in state input.
    """

    def __init__(
        self,
        *args,
        interval_embed_dim: int = 64,
        interval_hidden_dim: int = 128,
        momentum_meanflow: dict | None = None,
        momentum_meanflow_schedule: str | None = None,
        state_input_multiplier: int = 2,
        **kwargs,
    ) -> None:
        cfg = momentum_meanflow or {}
        self.mm_cfg = MomentumMeanFlowConfig(
            interval_embed_dim=int(cfg.get("interval_embed_dim", interval_embed_dim)),
            interval_hidden_dim=int(cfg.get("interval_hidden_dim", interval_hidden_dim)),
            lambda_correct=float(cfg.get("lambda_correct", 1.0)),
            dt_min=float(cfg.get("dt_min", 1.0 / 32.0)),
            momentum_meanflow_schedule=str(
                momentum_meanflow_schedule
                or cfg.get("momentum_meanflow_schedule", "uniform")
            ),
        )
        if self.mm_cfg.momentum_meanflow_schedule not in MOMENTUM_INFER_SCHEDULES:
            raise ValueError(
                f"Unknown momentum_meanflow_schedule={self.mm_cfg.momentum_meanflow_schedule}. "
                f"Expected one of {MOMENTUM_INFER_SCHEDULES}"
            )
        super().__init__(
            *args,
            state_input_multiplier=int(state_input_multiplier),
            **kwargs,
        )
        self.state_input_multiplier = int(state_input_multiplier)
        self.sampler_mode = "momentum_meanflow_multistep"
        self.meanflow_nfe: int = 0
        self.interval_mlp = nn.Sequential(
            nn.Linear(5, self.mm_cfg.interval_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.mm_cfg.interval_hidden_dim, self.mm_cfg.interval_embed_dim),
        )
        if int(self.num_k_infer) <= 0:
            self.num_k_infer = 1
        logger.info(
            "[MomentumMeanFlow] num_k_infer=%d schedule=%s lambda_correct=%.4f dt_min=%.6f "
            "state_input_multiplier=%d interval_embed_dim=%d",
            int(self.num_k_infer),
            self.mm_cfg.momentum_meanflow_schedule,
            self.mm_cfg.lambda_correct,
            self.mm_cfg.dt_min,
            int(state_input_multiplier),
            self.mm_cfg.interval_embed_dim,
        )
        print(
            "[MomentumMeanFlow] "
            f"num_k_infer={int(self.num_k_infer)} "
            f"schedule={self.mm_cfg.momentum_meanflow_schedule} "
            f"lambda_correct={self.mm_cfg.lambda_correct} "
            f"dt_min={self.mm_cfg.dt_min} "
            f"state_input_multiplier={int(state_input_multiplier)}"
        )

    def set_num_k_infer(self, num_k_infer: int) -> None:
        k = int(num_k_infer)
        if k <= 0:
            raise ValueError("num_k_infer must be >= 1")
        self.num_k_infer = k
        return

    def set_momentum_meanflow_schedule(self, schedule: str) -> None:
        schedule = str(schedule)
        if schedule not in MOMENTUM_INFER_SCHEDULES:
            raise ValueError(f"Unknown momentum_meanflow schedule: {schedule}")
        self.mm_cfg.momentum_meanflow_schedule = schedule
        return

    def _momentum_global_cond(
        self,
        nx: torch.Tensor,
        t_prev: torch.Tensor,
        t_cur: torch.Tensor,
        t_next: torch.Tensor,
    ) -> torch.Tensor:
        b = nx.shape[0]
        dt_prev = t_cur - t_prev
        dt_next = t_next - t_cur
        interval_feat = torch.cat(
            [
                t_prev.reshape(b, 1),
                t_cur.reshape(b, 1),
                t_next.reshape(b, 1),
                dt_prev.reshape(b, 1),
                dt_next.reshape(b, 1),
            ],
            dim=-1,
        )
        interval_emb = self.interval_mlp(interval_feat)
        return torch.cat([nx, interval_emb], dim=-1)

    def _build_state_input(
        self,
        x_current: torch.Tensor,
        prev_u: torch.Tensor,
        phase_flow: torch.Tensor | None,
    ) -> torch.Tensor:
        x_in = torch.cat([x_current, prev_u], dim=-1)
        if self.phase_enabled:
            assert phase_flow is not None
            phase_emb = self.phase_embedding(phase_flow)
            x_in = torch.cat([x_in, phase_emb], dim=-1)
        return x_in

    def _forward_u(
        self,
        nx: torch.Tensor,
        x_current: torch.Tensor,
        prev_u: torch.Tensor,
        t_prev: torch.Tensor,
        t_cur: torch.Tensor,
        t_next: torch.Tensor,
        *,
        phase_flow: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b = x_current.shape[0]
        x_in = self._build_state_input(x_current, prev_u, phase_flow)
        timesteps = t_cur.reshape(b) * self.pos_emb_scale if self.time_conditioning else None
        global_cond = self._momentum_global_cond(nx, t_prev, t_cur, t_next)
        pred_full = self.diffusion_net(x_in, timesteps, global_cond=global_cond)
        return self._slice_velocity(pred_full)

    def _weighted_velocity_loss(
        self,
        pred_u: torch.Tensor,
        target_u: torch.Tensor,
        ny: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        per_xyz, per_rot6d, per_grip = self._per_timestep_loss(pred_u, target_u)
        weights = self._compute_gripper_weights(ny)
        loss_xyz = (per_xyz * weights).mean()
        loss_rot6d = (per_rot6d * weights).mean()
        loss_grip = (per_grip * weights).mean()
        return loss_xyz, loss_rot6d, loss_grip

    def _compute_losses(
        self,
        pcd: torch.Tensor,
        robot_state_obs: torch.Tensor,
        robot_state_pred: torch.Tensor,
        *,
        phase: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
        nx = self.encode_obs(pcd, robot_state_obs)
        ny = robot_state_pred
        b, _, d = ny.shape
        device = ny.device

        phase_flow = None
        phase_metrics: dict[str, float] = {}
        if self.phase_enabled:
            if phase is None:
                raise ValueError("phase_conditioning.enabled=true requires phase labels.")
            phase = self._assert_valid_phase(phase, b, ny.shape[1])
            phase_flow = phase
            if self.phase_pred_enabled:
                _, phase_metrics = self._compute_phase_aux_loss(nx, phase)
                phase_flow = self._phase_for_flow_train(phase, nx, B=b, T=ny.shape[1])
            phase_metrics.update(self._phase_flow_train_metrics(phase_flow, phase))

        z0 = self._init_noise(b)
        z1 = ny
        dt_min = float(self.mm_cfg.dt_min)

        t0, t1 = sample_two_times(b, dt_min=dt_min, device=device)
        dt01 = torch.clamp(t1 - t0, min=dt_min)
        x_t0 = (1.0 - t0) * z0 + t0 * z1
        x_t1 = (1.0 - t1) * z0 + t1 * z1
        prev_u0 = torch.zeros_like(x_t0)
        u01_target = (x_t1 - x_t0) / dt01
        u01_pred = self._forward_u(
            nx,
            x_t0,
            prev_u0,
            t_prev=t0,
            t_cur=t0,
            t_next=t1,
            phase_flow=phase_flow,
        )
        l01_xyz, l01_rot6d, l01_grip = self._weighted_velocity_loss(u01_pred, u01_target, ny)

        t0c, t1c, t2c = sample_three_times(b, dt_min=dt_min, device=device)
        dt01c = torch.clamp(t1c - t0c, min=dt_min)
        dt12 = torch.clamp(t2c - t1c, min=dt_min)
        x_t0c = (1.0 - t0c) * z0 + t0c * z1
        x_t1 = (1.0 - t1c) * z0 + t1c * z1
        x_t2 = (1.0 - t2c) * z0 + t2c * z1

        u01_for_corr = self._forward_u(
            nx,
            x_t0c,
            torch.zeros_like(x_t0c),
            t_prev=t0c,
            t_cur=t0c,
            t_next=t1c,
            phase_flow=phase_flow,
        )
        u01_det = u01_for_corr.detach()
        x_t1_hat = (x_t0c + dt01c * u01_det).detach()
        u12_target = (x_t2 - x_t1_hat) / dt12
        u12_pred = self._forward_u(
            nx,
            x_t1_hat,
            u01_det,
            t_prev=t0c,
            t_cur=t1c,
            t_next=t2c,
            phase_flow=phase_flow,
        )
        l12_xyz, l12_rot6d, l12_grip = self._weighted_velocity_loss(u12_pred, u12_target, ny)

        lam = float(self.mm_cfg.lambda_correct)
        loss_xyz = l01_xyz + lam * l12_xyz
        loss_rot6d = l01_rot6d + lam * l12_rot6d
        loss_grip = l01_grip + lam * l12_grip

        stats = {
            "momentum_meanflow/lambda_correct": lam,
            "momentum_meanflow/dt01_mean": float(dt01.mean().item()),
            "momentum_meanflow/dt12_mean": float(dt12.mean().item()),
            "momentum_meanflow/u01_norm": float(u01_pred.detach().norm(dim=-1).mean().item()),
            "momentum_meanflow/u12_target_norm": float(u12_target.detach().norm(dim=-1).mean().item()),
            "momentum_meanflow/u12_pred_norm": float(u12_pred.detach().norm(dim=-1).mean().item()),
            "momentum_meanflow/correction_error_norm": float((x_t1_hat - x_t1).detach().norm(dim=-1).mean().item()),
            "loss/train/momentum_first_xyz": float(l01_xyz.item()),
            "loss/train/momentum_first_rot6d": float(l01_rot6d.item()),
            "loss/train/momentum_first_grip": float(l01_grip.item()),
            "loss/train/momentum_correct_xyz": float(l12_xyz.item()),
            "loss/train/momentum_correct_rot6d": float(l12_rot6d.item()),
            "loss/train/momentum_correct_grip": float(l12_grip.item()),
            "momentum_meanflow/num_k_infer": float(self.num_k_infer),
            "momentum_meanflow/schedule": self.mm_cfg.momentum_meanflow_schedule,
        }
        stats.update(phase_metrics)
        return loss_xyz, loss_rot6d, loss_grip, stats

    def loss(self, outputs, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
        with torch.no_grad():
            batch = self._norm_data(batch)
            if self.augment_data:
                batch = self._augment_data(batch)
        pcd, robot_state_obs, robot_state_pred = batch[:3]
        phase_pred = batch[3] if (self.phase_enabled and len(batch) >= 4) else None
        loss_xyz, loss_rot6d, loss_grip, stats = self._compute_losses(
            pcd, robot_state_obs, robot_state_pred, phase=phase_pred
        )
        cfm_loss = (
            self.l_w["xyz"] * loss_xyz + self.l_w["rot6d"] * loss_rot6d + self.l_w["grip"] * loss_grip
        )
        metrics = {
            "loss/train/xyz": loss_xyz.item(),
            "loss/train/rot6d": loss_rot6d.item(),
            "loss/train/grip": loss_grip.item(),
            "loss/train/cfm": cfm_loss.item(),
            "loss/train/total": cfm_loss.item(),
        }
        metrics.update(stats)
        self.logger.log_metrics(metrics)
        return cfm_loss

    def calculate_loss(
        self,
        pcd: torch.Tensor,
        robot_state_obs: torch.Tensor,
        robot_state_pred: torch.Tensor,
        *,
        phase: torch.Tensor | None = None,
    ):
        loss_xyz, loss_rot6d, loss_grip, _ = self._compute_losses(
            pcd, robot_state_obs, robot_state_pred, phase=phase
        )
        return loss_xyz, loss_rot6d, loss_grip, None, {}

    def infer_y(
        self,
        pcd: torch.Tensor,
        robot_state_obs: torch.Tensor,
        phase: torch.Tensor | None = None,
        noise=None,
        return_traj=False,
    ) -> torch.Tensor:
        t0 = time.perf_counter()
        self.meanflow_nfe = 0
        yprof = blank_infer_y_profile() if self.profile_inference else None
        if yprof is not None:
            cuda_sync()
            t_enc0 = time.perf_counter()
            nx = self.encode_obs(pcd, robot_state_obs)
            cuda_sync()
            yprof["encode_obs_ms"] += (time.perf_counter() - t_enc0) * 1000.0
        else:
            nx = self.encode_obs(pcd, robot_state_obs)

        b = nx.shape[0]
        z = self._init_noise(b) if noise is None else noise
        phase_flow = None
        if self.phase_enabled:
            if phase is None:
                phase_flow = torch.zeros((b, self.n_pred_steps), device=DEVICE, dtype=torch.int64)
            else:
                if phase.dtype != torch.int64:
                    phase = phase.to(torch.int64)
                phase_flow = phase.view(b, 1).repeat(1, self.n_pred_steps) if phase.ndim == 1 else phase

        k = int(self.num_k_infer)
        if k <= 0:
            raise ValueError("num_k_infer must be >= 1")

        grid = build_inference_time_grid(
            k,
            self.mm_cfg.momentum_meanflow_schedule,
            exp_scale=float(self.exp_scale or 4.0),
            device=z.device,
            dtype=z.dtype,
        )
        prev_u = torch.zeros_like(z)
        traj = [z]
        if yprof is not None:
            cuda_sync()
            loop_t0 = time.perf_counter()

        for i in range(k):
            if i == 0:
                t_prev_v = float(grid[0])
                t_cur_v = float(grid[0])
            else:
                t_prev_v = float(grid[i - 1])
                t_cur_v = float(grid[i])
            t_next_v = float(grid[i + 1])
            t_prev = torch.full((b, 1, 1), t_prev_v, device=z.device, dtype=z.dtype)
            t_cur = torch.full((b, 1, 1), t_cur_v, device=z.device, dtype=z.dtype)
            t_next = torch.full((b, 1, 1), t_next_v, device=z.device, dtype=z.dtype)
            dt_next = t_next - t_cur
            if yprof is not None:
                cuda_sync()
                t_mlp0 = time.perf_counter()
                u = self._forward_u(
                    nx, z, prev_u, t_prev=t_prev, t_cur=t_cur, t_next=t_next, phase_flow=phase_flow
                )
                cuda_sync()
                yprof["extra_mlp_ms"] += (time.perf_counter() - t_mlp0) * 1000.0
                yprof["unet_total_ms"] += (time.perf_counter() - t_mlp0) * 1000.0
                yprof["nfe"] += 1.0
            else:
                u = self._forward_u(
                    nx, z, prev_u, t_prev=t_prev, t_cur=t_cur, t_next=t_next, phase_flow=phase_flow
                )
            self.meanflow_nfe += 1
            z = z + dt_next * u
            prev_u = u
            traj.append(z)

        if yprof is not None:
            cuda_sync()
            yprof["loop_total_ms"] += (time.perf_counter() - loop_t0) * 1000.0
            self._last_infer_y_profile = yprof

        self.last_infer_nfe = int(self.meanflow_nfe)
        self._infer_calls_total += 1
        self._infer_actions_total += int(b)
        self._infer_nfe_total += int(self.meanflow_nfe) * int(b)
        self._infer_time_total_ms += float((time.perf_counter() - t0) * 1000.0)
        if return_traj:
            return torch.stack(traj)
        return traj[-1]
