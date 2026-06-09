from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from pfp import DEVICE
from pfp.policy.fm_policy import FMPolicy

logger = logging.getLogger(__name__)


@dataclass
class StepConditionedMeanFlowConfig:
    embed_dim: int = 128
    hidden_dim: int = 256
    train_delta_steps: tuple[int, ...] = (1, 2, 4, 8, 16)
    use_consistency_loss: bool = True
    lambda_consistency: float = 0.1
    stopgrad_target: bool = True


class StepConditionedMeanFlowPolicy(FMPolicy):
    """
    Unified step-conditioned interval-flow model.

    Predicts:
      u_theta(x_t, t, delta, cond)
    and integrates with:
      x_{t+delta} = x_t + delta * u_theta(...)
    """

    def __init__(
        self,
        *args,
        step_conditioned_meanflow: dict | None = None,
        num_inference_steps: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        cfg = step_conditioned_meanflow or {}
        raw_steps = cfg.get("train_delta_steps", [1, 2, 4, 8, 16])
        steps = tuple(sorted({int(s) for s in raw_steps if int(s) > 0}))
        if not steps:
            raise ValueError("step_conditioned_meanflow.train_delta_steps must contain positive integers")
        self.scmf_cfg = StepConditionedMeanFlowConfig(
            embed_dim=int(cfg.get("embed_dim", 128)),
            hidden_dim=int(cfg.get("hidden_dim", 256)),
            train_delta_steps=steps,
            use_consistency_loss=bool(cfg.get("use_consistency_loss", True)),
            lambda_consistency=float(cfg.get("lambda_consistency", 0.1)),
            stopgrad_target=bool(cfg.get("stopgrad_target", True)),
        )
        if num_inference_steps is not None:
            self.num_k_infer = int(num_inference_steps)
        if self.num_k_infer <= 0:
            raise ValueError("num_k_infer must be >= 1")

        gc_dim = int(self.x_dim) * int(self.n_obs_steps)
        self.t_embed = nn.Sequential(
            nn.Linear(1, self.scmf_cfg.embed_dim),
            nn.ReLU(),
            nn.Linear(self.scmf_cfg.embed_dim, self.scmf_cfg.embed_dim),
        )
        self.delta_embed = nn.Sequential(
            nn.Linear(1, self.scmf_cfg.embed_dim),
            nn.ReLU(),
            nn.Linear(self.scmf_cfg.embed_dim, self.scmf_cfg.embed_dim),
        )
        self.step_cond_mlp = nn.Sequential(
            nn.Linear(gc_dim + 2 * self.scmf_cfg.embed_dim, self.scmf_cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.scmf_cfg.hidden_dim, self.scmf_cfg.embed_dim),
        )
        logger.info(
            "[SCMF] num_k_infer=%d train_delta_steps=%s use_consistency=%s lambda_consistency=%.4f stopgrad=%s",
            int(self.num_k_infer),
            list(self.scmf_cfg.train_delta_steps),
            self.scmf_cfg.use_consistency_loss,
            self.scmf_cfg.lambda_consistency,
            self.scmf_cfg.stopgrad_target,
        )
        print(
            "[SCMF] "
            f"num_k_infer={int(self.num_k_infer)} "
            f"train_delta_steps={list(self.scmf_cfg.train_delta_steps)} "
            f"use_consistency_loss={self.scmf_cfg.use_consistency_loss} "
            f"lambda_consistency={self.scmf_cfg.lambda_consistency} "
            f"stopgrad_target={self.scmf_cfg.stopgrad_target}"
        )

    def set_num_k_infer(self, num_k_infer: int):
        k = int(num_k_infer)
        if k <= 0:
            raise ValueError("num_k_infer must be >= 1")
        self.num_k_infer = k
        return

    def _infer_phase_seq(
        self,
        nx: torch.Tensor,
        phase: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if not self.phase_enabled:
            return None
        b = nx.shape[0]
        if phase is None:
            if self.phase_pred_enabled:
                phase_logits = self.predict_phase_logits(nx)
                phase_current = phase_logits.argmax(dim=-1)
                phase_seq = phase_current.view(b, 1).expand(b, self.n_pred_steps)
            else:
                phase_seq = torch.zeros((b, self.n_pred_steps), device=nx.device, dtype=torch.int64)
        else:
            if phase.dtype != torch.int64:
                phase = phase.to(torch.int64)
            phase_seq = phase.view(b, 1).repeat(1, self.n_pred_steps) if phase.ndim == 1 else phase
        if phase_seq.shape != (b, self.n_pred_steps):
            raise ValueError(f"phase must broadcast to (B,T)={(b, self.n_pred_steps)}, got {tuple(phase_seq.shape)}")
        return phase_seq

    def _sample_delta_and_t(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        levels = torch.tensor(
            [1.0 / float(s) for s in self.scmf_cfg.train_delta_steps],
            device=device,
            dtype=torch.float32,
        )
        idx = torch.randint(0, levels.shape[0], size=(batch_size,), device=device)
        delta = levels[idx].view(batch_size, 1, 1)
        t_max = torch.clamp(1.0 - delta, min=0.0)
        t = torch.rand((batch_size, 1, 1), device=device) * t_max
        return delta, t

    def _step_global_cond(self, nx: torch.Tensor, t: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        b = nx.shape[0]
        t_emb = self.t_embed(t.view(b, 1))
        d_emb = self.delta_embed(delta.view(b, 1))
        step_cond = self.step_cond_mlp(torch.cat([nx, t_emb, d_emb], dim=-1))
        return torch.cat([nx, step_cond], dim=-1)

    def _forward_step_conditioned(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        delta: torch.Tensor,
        nx: torch.Tensor,
        *,
        phase_flow: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b = x_t.shape[0]
        if self.phase_enabled:
            assert phase_flow is not None
            phase_emb = self.phase_embedding(phase_flow)
            x_in = torch.cat([x_t, phase_emb], dim=-1)
        else:
            x_in = x_t
        timesteps = t.view(b) * self.pos_emb_scale if self.time_conditioning else None
        global_cond = self._step_global_cond(nx, t, delta)
        pred_full = self.diffusion_net(x_in, timesteps, global_cond=global_cond)
        return self._slice_velocity(pred_full)

    def _compute_losses(
        self,
        pcd: torch.Tensor,
        robot_state_obs: torch.Tensor,
        robot_state_pred: torch.Tensor,
        *,
        phase: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
        nx = self.encode_obs(pcd, robot_state_obs)
        ny = robot_state_pred
        b, horizon, _ = ny.shape
        device = ny.device

        phase_flow = None
        phase_loss = None
        phase_metrics: dict[str, float] = {}
        if self.phase_enabled:
            if phase is None:
                raise ValueError("phase_conditioning.enabled=true requires phase labels in SCMF loss.")
            phase = self._assert_valid_phase(phase, b, horizon)
            phase_flow = phase
            if self.phase_pred_enabled:
                phase_loss, phase_metrics = self._compute_phase_aux_loss(nx, phase)
                phase_flow = self._phase_for_flow_train(phase, nx, B=b, T=horizon)
            phase_metrics.update(self._phase_flow_train_metrics(phase_flow, phase))

        z0 = self._init_noise(b)
        z1 = ny

        delta, t = self._sample_delta_and_t(b, device)
        x_t = (1.0 - t) * z0 + t * z1
        t_next = t + delta
        x_next = (1.0 - t_next) * z0 + t_next * z1
        target_u = (x_next - x_t) / torch.clamp(delta, min=1e-8)

        pred_u = self._forward_step_conditioned(x_t, t, delta, nx, phase_flow=phase_flow)
        per_xyz, per_rot6d, per_grip = self._per_timestep_loss(pred_u, target_u)
        weights = self._compute_gripper_weights(ny)
        loss_flow_xyz = (per_xyz * weights).mean()
        loss_flow_rot6d = (per_rot6d * weights).mean()
        loss_flow_grip = (per_grip * weights).mean()

        loss_consistency = torch.zeros((), device=device, dtype=ny.dtype)
        if self.scmf_cfg.use_consistency_loss:
            half = 0.5 * delta
            u_big = self._forward_step_conditioned(x_t, t, delta, nx, phase_flow=phase_flow)
            x_big = x_t + delta * u_big

            u_1 = self._forward_step_conditioned(x_t, t, half, nx, phase_flow=phase_flow)
            x_mid = x_t + half * u_1
            u_2 = self._forward_step_conditioned(x_mid, t + half, half, nx, phase_flow=phase_flow)
            x_two = x_mid + half * u_2

            target_two = x_two.detach() if self.scmf_cfg.stopgrad_target else x_two
            loss_consistency = F.mse_loss(x_big, target_two)

        stats = {
            "step_conditioned/delta_mean": float(delta.mean().item()),
            "step_conditioned/delta_min": float(delta.min().item()),
            "step_conditioned/delta_max": float(delta.max().item()),
            "step_conditioned/num_steps_mean": float((1.0 / delta).mean().item()),
            "step_conditioned/t_mean": float(t.mean().item()),
            "step_conditioned/use_consistency_loss": 1.0 if self.scmf_cfg.use_consistency_loss else 0.0,
            "step_conditioned/lambda_consistency": float(self.scmf_cfg.lambda_consistency),
            "step_conditioned/num_k_infer": float(self.num_k_infer),
        }
        if phase_loss is not None:
            stats.update(phase_metrics)
        return loss_flow_xyz, loss_flow_rot6d, loss_flow_grip, loss_consistency, stats

    def loss(self, outputs, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
        with torch.no_grad():
            batch = self._norm_data(batch)
            if self.augment_data:
                batch = self._augment_data(batch)
        pcd, robot_state_obs, robot_state_pred = batch[:3]
        phase_pred = batch[3] if (self.phase_enabled and len(batch) >= 4) else None
        flow_xyz, flow_rot6d, flow_grip, loss_consistency, stats = self._compute_losses(
            pcd, robot_state_obs, robot_state_pred, phase=phase_pred
        )
        loss_flow = (
            self.l_w["xyz"] * flow_xyz
            + self.l_w["rot6d"] * flow_rot6d
            + self.l_w["grip"] * flow_grip
        )
        total = loss_flow + self.scmf_cfg.lambda_consistency * loss_consistency
        metrics = {
            "loss/train/flow_xyz": float(flow_xyz.item()),
            "loss/train/flow_rot6d": float(flow_rot6d.item()),
            "loss/train/flow_grip": float(flow_grip.item()),
            "loss/train/loss_flow": float(loss_flow.item()),
            "loss/train/loss_consistency": float(loss_consistency.item()),
            "loss/train/total": float(total.item()),
            **stats,
        }
        self.logger.log_metrics(metrics)
        return total

    def eval_forward(self, batch: tuple[torch.Tensor, ...], outputs=None) -> torch.Tensor:
        batch = self._norm_data(batch)
        pcd, robot_state_obs, robot_state_pred = batch[:3]
        phase_pred = batch[3] if (self.phase_enabled and len(batch) >= 4) else None
        flow_xyz, flow_rot6d, flow_grip, loss_consistency, stats = self._compute_losses(
            pcd, robot_state_obs, robot_state_pred, phase=phase_pred
        )
        loss_flow = (
            self.l_w["xyz"] * flow_xyz
            + self.l_w["rot6d"] * flow_rot6d
            + self.l_w["grip"] * flow_grip
        )
        total = loss_flow + self.scmf_cfg.lambda_consistency * loss_consistency
        eval_metrics = {
            "loss/eval/flow_xyz": float(flow_xyz.item()),
            "loss/eval/flow_rot6d": float(flow_rot6d.item()),
            "loss/eval/flow_grip": float(flow_grip.item()),
            "loss/eval/loss_flow": float(loss_flow.item()),
            "loss/eval/loss_consistency": float(loss_consistency.item()),
            "loss/eval/total": float(total.item()),
            "step_conditioned/eval_delta_mean": stats["step_conditioned/delta_mean"],
            "step_conditioned/eval_delta_min": stats["step_conditioned/delta_min"],
            "step_conditioned/eval_delta_max": stats["step_conditioned/delta_max"],
            "step_conditioned/eval_num_steps_mean": stats["step_conditioned/num_steps_mean"],
        }
        self.logger.log_metrics(eval_metrics)

        infer_phase = None if self.phase_pred_enabled else phase_pred
        pred_y = self.infer_y(pcd, robot_state_obs, phase=infer_phase)
        mse_xyz = nn.functional.mse_loss(pred_y[..., :3], robot_state_pred[..., :3])
        mse_rot6d = nn.functional.mse_loss(pred_y[..., 3:9], robot_state_pred[..., 3:9])
        mse_grip = nn.functional.mse_loss(pred_y[..., 9], robot_state_pred[..., 9])
        self.logger.log_metrics(
            {
                "metrics/eval/mse_xyz": mse_xyz.item(),
                "metrics/eval/mse_rot6d": mse_rot6d.item(),
                "metrics/eval/mse_grip": mse_grip.item(),
            }
        )
        return pred_y

    def infer_y(
        self,
        pcd: torch.Tensor,
        robot_state_obs: torch.Tensor,
        phase: torch.Tensor | None = None,
        noise=None,
        return_traj=False,
    ) -> torch.Tensor:
        infer_t0 = time.perf_counter()
        nx = self.encode_obs(pcd, robot_state_obs)
        b = nx.shape[0]
        z = self._init_noise(b) if noise is None else noise
        traj = [z]
        phase_seq = self._infer_phase_seq(nx, phase)

        k = int(self.num_k_infer)
        if k <= 0:
            raise ValueError("num_k_infer must be >= 1")
        delta_val = 1.0 / float(k)
        delta = torch.full((b, 1, 1), delta_val, device=DEVICE, dtype=z.dtype)

        for i in range(k):
            t = torch.full((b, 1, 1), float(i) * delta_val, device=DEVICE, dtype=z.dtype)
            u = self._forward_step_conditioned(z, t, delta, nx, phase_flow=phase_seq)
            z = z + delta * u
            traj.append(z)

        infer_ms = (time.perf_counter() - infer_t0) * 1000.0
        self.last_infer_nfe = int(k)
        self._infer_calls_total += 1
        self._infer_actions_total += int(b)
        self._infer_nfe_total += int(k) * int(b)
        self._infer_time_total_ms += float(infer_ms)

        if return_traj:
            return torch.stack(traj)
        return traj[-1]
