from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from pfp import DEVICE
from pfp.common.inference_profiling import blank_infer_y_profile, cuda_sync
from pfp.policy.fm_policy import FMPolicy

logger = logging.getLogger(__name__)


@dataclass
class ShortcutConfig:
    embed_dim: int = 128
    num_base_steps: int = 32
    base_loss_weight: float = 1.0
    consistency_loss_weight: float = 1.0
    self_consistency_fraction: float = 0.5
    stopgrad_target: bool = True
    include_one_step_target: bool = True
    one_step_loss_weight: float = 1.0
    interpolation_eps: float = 1e-3
    state_clamp_value: float = 4.0


class ShortcutFMPolicy(FMPolicy):
    """
    Shortcut PointFlowMatch:
      s_theta(x_t, t, d, obs), with explicit step-size conditioning d.
    """

    def __init__(self, *args, shortcut: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = shortcut or {}
        self.shortcut_cfg = ShortcutConfig(
            embed_dim=int(cfg.get("embed_dim", 128)),
            num_base_steps=int(cfg.get("num_base_steps", 8)),
            base_loss_weight=float(cfg.get("base_loss_weight", 1.0)),
            consistency_loss_weight=float(cfg.get("consistency_loss_weight", 1.0)),
            self_consistency_fraction=float(cfg.get("self_consistency_fraction", 0.5)),
            stopgrad_target=bool(cfg.get("stopgrad_target", True)),
            include_one_step_target=bool(cfg.get("include_one_step_target", True)),
            one_step_loss_weight=float(cfg.get("one_step_loss_weight", 1.0)),
            interpolation_eps=float(cfg.get("interpolation_eps", 1e-3)),
            state_clamp_value=float(cfg.get("state_clamp_value", 4.0)),
        )
        if self.shortcut_cfg.num_base_steps <= 1:
            raise ValueError("shortcut.num_base_steps must be > 1")
        if self.shortcut_cfg.num_base_steps & (self.shortcut_cfg.num_base_steps - 1):
            raise ValueError("shortcut.num_base_steps must be a power of 2 so d_min=1/num_base_steps is dyadic")
        if not 0.0 <= self.shortcut_cfg.self_consistency_fraction <= 1.0:
            raise ValueError("shortcut.self_consistency_fraction must be in [0, 1]")

        self.d_embed = nn.Sequential(
            nn.Linear(1, self.shortcut_cfg.embed_dim),
            nn.ReLU(),
            nn.Linear(self.shortcut_cfg.embed_dim, self.shortcut_cfg.embed_dim),
        )
        logger.info(
            "[ShortcutFlow] embed_dim=%d num_base_steps=%d base_w=%.3f cons_w=%.3f "
            "self_consistency_fraction=%.3f one_step_w=%.3f include_one_step=%s stopgrad=%s",
            self.shortcut_cfg.embed_dim,
            self.shortcut_cfg.num_base_steps,
            self.shortcut_cfg.base_loss_weight,
            self.shortcut_cfg.consistency_loss_weight,
            self.shortcut_cfg.self_consistency_fraction,
            self.shortcut_cfg.one_step_loss_weight,
            self.shortcut_cfg.include_one_step_target,
            self.shortcut_cfg.stopgrad_target,
        )
        print(
            f"[ShortcutFlow] embed_dim={self.shortcut_cfg.embed_dim} "
            f"num_base_steps={self.shortcut_cfg.num_base_steps} "
            f"base_w={self.shortcut_cfg.base_loss_weight} cons_w={self.shortcut_cfg.consistency_loss_weight} "
            f"self_consistency_fraction={self.shortcut_cfg.self_consistency_fraction} "
            f"one_step_w={self.shortcut_cfg.one_step_loss_weight} "
            f"include_one_step_target={self.shortcut_cfg.include_one_step_target} "
            f"stopgrad_target={self.shortcut_cfg.stopgrad_target} "
            f"state_clamp_value={self.shortcut_cfg.state_clamp_value}"
        )

    def _augment_global_cond_with_d(self, nx: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        b = nx.shape[0]
        d_in = d.view(b, 1)
        d_emb = self.d_embed(d_in)
        return torch.cat([nx, d_emb], dim=-1)

    def _interpolate_shortcut_state(self, z0: torch.Tensor, z1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        eps = float(self.shortcut_cfg.interpolation_eps)
        return self._clamp_shortcut_state((1.0 - (1.0 - eps) * t) * z0 + t * z1)

    def _shortcut_target_velocity(self, z0: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
        eps = float(self.shortcut_cfg.interpolation_eps)
        return z1 - (1.0 - eps) * z0

    def _clamp_shortcut_state(self, x: torch.Tensor) -> torch.Tensor:
        c = float(self.shortcut_cfg.state_clamp_value)
        return torch.clamp(x, min=-c, max=c)

    def _forward_shortcut(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        d: torch.Tensor,
        nx: torch.Tensor,
        phase_flow: torch.Tensor | None = None,
        yprof: dict[str, float] | None = None,
    ) -> torch.Tensor:
        b = x_t.shape[0]
        if self.phase_enabled:
            assert phase_flow is not None
            phase_emb = self.phase_embedding(phase_flow)
            x_in = torch.cat([x_t, phase_emb], dim=-1)
        else:
            x_in = x_t
        timesteps = t.view(b) * self.pos_emb_scale if self.time_conditioning else None
        if yprof is not None:
            cuda_sync()
            t_mlp0 = time.perf_counter()
            global_cond = self._augment_global_cond_with_d(nx, d)
            cuda_sync()
            yprof["extra_mlp_ms"] += (time.perf_counter() - t_mlp0) * 1000.0
            cuda_sync()
            t_unet0 = time.perf_counter()
            pred_full = self.diffusion_net(x_in, timesteps, global_cond=global_cond)
            cuda_sync()
            yprof["unet_total_ms"] += (time.perf_counter() - t_unet0) * 1000.0
            yprof["nfe"] += 1.0
        else:
            global_cond = self._augment_global_cond_with_d(nx, d)
            pred_full = self.diffusion_net(x_in, timesteps, global_cond=global_cond)
        pred = self._slice_velocity(pred_full)
        return pred

    def _compute_shortcut_losses(
        self,
        pcd: torch.Tensor,
        robot_state_obs: torch.Tensor,
        robot_state_pred: torch.Tensor,
        *,
        phase: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
        nx = self.encode_obs(pcd, robot_state_obs)
        ny = robot_state_pred
        b, t_h, _ = ny.shape
        phase_flow = None
        if self.phase_enabled:
            if phase is None:
                raise ValueError("phase_conditioning.enabled=true requires phase labels in shortcut loss.")
            phase = self._assert_valid_phase(phase, b, t_h)
            phase_flow = phase

        d_min_value = 1.0 / float(self.shortcut_cfg.num_base_steps)
        assert self.shortcut_cfg.num_base_steps > 0
        assert self.shortcut_cfg.num_base_steps & (self.shortcut_cfg.num_base_steps - 1) == 0, (
            "shortcut d_min must be a power-of-two reciprocal"
        )

        n_sc = int(round(b * float(self.shortcut_cfg.self_consistency_fraction)))
        n_sc = max(0, min(b, n_sc))
        if 0.0 < self.shortcut_cfg.self_consistency_fraction < 1.0 and b > 1:
            n_sc = max(1, min(b - 1, n_sc))
        n_base = b - n_sc

        zero = ny.sum() * 0.0
        loss_base_xyz = zero
        loss_base_rot6d = zero
        loss_base_grip = zero
        loss_sc = zero
        loss_one_step = zero
        one_step_anchor_t_mean = 0.0

        d_sc = torch.empty((0, 1, 1), device=DEVICE, dtype=ny.dtype)
        t_sc = torch.empty((0, 1, 1), device=DEVICE, dtype=ny.dtype)

        if n_base > 0:
            nx_base = nx[:n_base]
            ny_base = ny[:n_base]
            phase_base = phase_flow[:n_base] if phase_flow is not None else None
            z0_base = self._init_noise(n_base)
            z1_base = ny_base

            t_base_idx = torch.randint(
                low=0,
                high=int(self.shortcut_cfg.num_base_steps),
                size=(n_base, 1, 1),
                device=DEVICE,
            )
            t_base = t_base_idx.to(dtype=ny.dtype) * d_min_value
            d_base = torch.full((n_base, 1, 1), d_min_value, device=DEVICE, dtype=ny.dtype)
            x_t = self._interpolate_shortcut_state(z0_base, z1_base, t_base)
            v_target = self._shortcut_target_velocity(z0_base, z1_base)
            v_pred = self._forward_shortcut(x_t, t_base, d_base, nx_base, phase_flow=phase_base)
            per_xyz, per_rot6d, per_grip = self._per_timestep_loss(v_pred, v_target)
            weights = self._compute_gripper_weights(ny_base)
            loss_base_xyz = (per_xyz * weights).mean()
            loss_base_rot6d = (per_rot6d * weights).mean()
            loss_base_grip = (per_grip * weights).mean()

        if n_sc > 0:
            nx_sc = nx[n_base:]
            ny_sc = ny[n_base:]
            phase_sc = phase_flow[n_base:] if phase_flow is not None else None
            z0_sc = self._init_noise(n_sc)
            z1_sc = ny_sc

            d_levels = []
            d_level = d_min_value
            while d_level <= 0.5 + 1e-12:
                d_levels.append(d_level)
                d_level *= 2.0
            levels = torch.tensor(d_levels, device=DEVICE, dtype=ny.dtype)
            d_weights = 1.0 / levels
            d_idx = torch.multinomial(d_weights, num_samples=n_sc, replacement=True)
            d_sc = levels[d_idx].view(n_sc, 1, 1)

            # For each d, sample t uniformly from the grid 0, 2d, 4d, ..., 1 - 2d.
            num_t_values = torch.round(1.0 / (2.0 * d_sc)).to(dtype=torch.long).view(n_sc)
            rand_unit = torch.rand((n_sc,), device=DEVICE)
            t_idx = torch.floor(rand_unit * num_t_values.to(dtype=ny.dtype)).to(dtype=ny.dtype).view(n_sc, 1, 1)
            t_sc = 2.0 * d_sc * t_idx

            x_sc = self._interpolate_shortcut_state(z0_sc, z1_sc, t_sc)
            v_big = self._forward_shortcut(x_sc, t_sc, 2.0 * d_sc, nx_sc, phase_flow=phase_sc)
            x_big = self._clamp_shortcut_state(x_sc + 2.0 * d_sc * v_big)

            v1 = self._forward_shortcut(x_sc, t_sc, d_sc, nx_sc, phase_flow=phase_sc)
            x_mid = self._clamp_shortcut_state(x_sc + d_sc * v1)
            v2 = self._forward_shortcut(x_mid, t_sc + d_sc, d_sc, nx_sc, phase_flow=phase_sc)
            x_two = self._clamp_shortcut_state(x_mid + d_sc * v2)

            x_two_target = x_two.detach() if self.shortcut_cfg.stopgrad_target else x_two
            loss_sc = F.mse_loss(x_big, x_two_target)

            if self.shortcut_cfg.include_one_step_target:
                # d=0.5 has a single valid shortcut grid anchor t=0 for the two half-steps.
                t_anchor = torch.zeros((n_sc, 1, 1), device=DEVICE, dtype=ny.dtype)
                x_anchor = self._interpolate_shortcut_state(z0_sc, z1_sc, t_anchor)
                d_full = torch.ones((n_sc, 1, 1), device=DEVICE, dtype=ny.dtype)
                v_full = self._forward_shortcut(x_anchor, t_anchor, d_full, nx_sc, phase_flow=phase_sc)
                x_full = self._clamp_shortcut_state(x_anchor + d_full * v_full)

                d_half = torch.full((n_sc, 1, 1), 0.5, device=DEVICE, dtype=ny.dtype)
                v_h1 = self._forward_shortcut(x_anchor, t_anchor, d_half, nx_sc, phase_flow=phase_sc)
                x_half = self._clamp_shortcut_state(x_anchor + d_half * v_h1)
                t_half1 = t_anchor + 0.5
                v_h2 = self._forward_shortcut(x_half, t_half1, d_half, nx_sc, phase_flow=phase_sc)
                x_half_half = self._clamp_shortcut_state(x_half + d_half * v_h2)
                x_half_half_target = x_half_half.detach() if self.shortcut_cfg.stopgrad_target else x_half_half
                loss_one_step = F.mse_loss(x_full, x_half_half_target)
                one_step_anchor_t_mean = float(t_anchor.mean().item())

        contains_half = bool(
            d_sc.numel() > 0 and torch.any(torch.isclose(d_sc.view(-1), torch.tensor(0.5, device=DEVICE))).item()
        )
        stats = {
            "shortcut/base_batch_size": float(n_base),
            "shortcut/self_consistency_batch_size": float(n_sc),
            "shortcut/self_consistency_fraction": float(self.shortcut_cfg.self_consistency_fraction),
            "shortcut/d_mean": float(d_sc.mean().item()) if d_sc.numel() > 0 else 0.0,
            "shortcut/t_mean": float(t_sc.mean().item()) if t_sc.numel() > 0 else 0.0,
            "shortcut/max_sampled_d": float(d_sc.max().item()) if d_sc.numel() > 0 else 0.0,
            "shortcut/contains_half_step": 1.0 if contains_half else 0.0,
            "shortcut/include_one_step_target": 1.0 if self.shortcut_cfg.include_one_step_target else 0.0,
            "shortcut/one_step_anchor_t_mean": one_step_anchor_t_mean,
            "shortcut/num_base_steps": float(self.shortcut_cfg.num_base_steps),
            "shortcut/d_min": float(d_min_value),
            "shortcut/state_clamp_value": float(self.shortcut_cfg.state_clamp_value),
            "shortcut/interpolation_eps": float(self.shortcut_cfg.interpolation_eps),
        }
        return loss_base_xyz, loss_base_rot6d, loss_base_grip, loss_sc, loss_one_step, stats

    def loss(self, outputs, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
        assert any(p.requires_grad for p in self.diffusion_net.parameters()), (
            "Shortcut diffusion_net parameters must require gradients"
        )
        with torch.no_grad():
            batch = self._norm_data(batch)
            if self.augment_data:
                batch = self._augment_data(batch)
        pcd, robot_state_obs, robot_state_pred = batch[:3]
        phase_pred = batch[3] if (self.phase_enabled and len(batch) >= 4) else None
        base_xyz, base_rot6d, base_grip, loss_sc, loss_one_step, sc_stats = self._compute_shortcut_losses(
            pcd, robot_state_obs, robot_state_pred, phase=phase_pred
        )

        base_5p = self.l_w["xyz"] * base_xyz + self.l_w["rot6d"] * base_rot6d
        base_grip_w = self.l_w["grip"] * base_grip
        l_base = base_5p + base_grip_w
        total = (
            self.shortcut_cfg.base_loss_weight * l_base
            + self.shortcut_cfg.consistency_loss_weight * loss_sc
            + self.shortcut_cfg.one_step_loss_weight * loss_one_step
        )
        assert total.requires_grad, "Shortcut total_loss must require gradients"
        self.logger.log_metrics(
            {
                "loss/train/base_5p": float(base_5p.item()),
                "loss/train/base_grip": float(base_grip_w.item()),
                "loss/train/shortcut_consistency": float(loss_sc.item()),
                "loss/train/shortcut_one_step": float(loss_one_step.item()),
                "loss/train/total": float(total.item()),
                **sc_stats,
            }
        )
        return total

    def eval_forward(self, batch: tuple[torch.Tensor, ...], outputs=None) -> torch.Tensor:
        batch = self._norm_data(batch)
        pcd, robot_state_obs, robot_state_pred = batch[:3]
        phase_pred = batch[3] if (self.phase_enabled and len(batch) >= 4) else None
        base_xyz, base_rot6d, base_grip, loss_sc, loss_one_step, sc_stats = self._compute_shortcut_losses(
            pcd, robot_state_obs, robot_state_pred, phase=phase_pred
        )
        base_5p = self.l_w["xyz"] * base_xyz + self.l_w["rot6d"] * base_rot6d
        base_grip_w = self.l_w["grip"] * base_grip
        l_base = base_5p + base_grip_w
        total = (
            self.shortcut_cfg.base_loss_weight * l_base
            + self.shortcut_cfg.consistency_loss_weight * loss_sc
            + self.shortcut_cfg.one_step_loss_weight * loss_one_step
        )
        self.logger.log_metrics(
            {
                "loss/eval/base_5p": float(base_5p.item()),
                "loss/eval/base_grip": float(base_grip_w.item()),
                "loss/eval/shortcut_consistency": float(loss_sc.item()),
                "loss/eval/shortcut_one_step": float(loss_one_step.item()),
                "loss/eval/total": float(total.item()),
                "shortcut/eval_d_mean": sc_stats["shortcut/d_mean"],
                "shortcut/eval_t_mean": sc_stats["shortcut/t_mean"],
                "shortcut/eval_max_sampled_d": sc_stats["shortcut/max_sampled_d"],
                "shortcut/eval_contains_half_step": sc_stats["shortcut/contains_half_step"],
            }
        )
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
        traj = [z]

        phase_seq = None
        if yprof is not None:
            cuda_sync()
            t_phase0 = time.perf_counter()
        if self.phase_enabled:
            if phase is None:
                if self.phase_pred_enabled:
                    phase_logits = self.predict_phase_logits(nx)
                    phase_current = phase_logits.argmax(dim=-1)
                    phase_seq = phase_current.view(b, 1).expand(b, self.n_pred_steps)
                else:
                    # default fallback when no phase is provided.
                    phase_seq = torch.zeros((b, self.n_pred_steps), device=DEVICE, dtype=torch.int64)
            else:
                if phase.dtype != torch.int64:
                    phase = phase.to(torch.int64)
                phase_seq = phase.view(b, 1).repeat(1, self.n_pred_steps) if phase.ndim == 1 else phase
            if phase_seq.shape != (b, self.n_pred_steps):
                raise ValueError(f"phase must broadcast to (B,T)={(b,self.n_pred_steps)}, got {tuple(phase_seq.shape)}")

        if yprof is not None:
            cuda_sync()
            yprof["other_loop_ms"] += (time.perf_counter() - t_phase0) * 1000.0

        k = int(self.num_k_infer)
        if k <= 0:
            raise ValueError("num_k_infer must be >= 1")

        if yprof is not None:
            cuda_sync()
            loop_t0 = time.perf_counter()
        if k == 1:
            d = torch.ones((b, 1, 1), device=DEVICE)
            t = torch.zeros((b, 1, 1), device=DEVICE)
            v = self._forward_shortcut(z, t, d, nx, phase_flow=phase_seq, yprof=yprof)
            z = z + d * v
            traj.append(z)
        else:
            d = torch.full((b, 1, 1), 1.0 / float(k), device=DEVICE)
            for i in range(k):
                t = torch.full((b, 1, 1), float(i) / float(k), device=DEVICE)
                v = self._forward_shortcut(z, t, d, nx, phase_flow=phase_seq, yprof=yprof)
                z = z + d * v
                traj.append(z)
        if yprof is not None:
            cuda_sync()
            yprof["loop_total_ms"] += (time.perf_counter() - loop_t0) * 1000.0
            self._last_infer_y_profile = yprof

        infer_ms = (time.perf_counter() - infer_t0) * 1000.0
        self.last_infer_nfe = int(k)
        self._infer_calls_total += 1
        self._infer_actions_total += int(b)
        self._infer_nfe_total += int(k) * int(b)
        self._infer_time_total_ms += float(infer_ms)

        if return_traj:
            return torch.stack(traj)
        return traj[-1]
