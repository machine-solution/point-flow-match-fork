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
class ShortcutConfig:
    embed_dim: int = 128
    num_base_steps: int = 10
    base_loss_weight: float = 1.0
    consistency_loss_weight: float = 1.0
    stopgrad_target: bool = True
    step_sampling: str = "powers_of_two"
    min_step: float | None = None
    max_step: float = 0.5
    include_half_step: bool = True
    include_one_step_target: bool = True
    one_step_loss_weight: float = 1.0


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
            num_base_steps=int(cfg.get("num_base_steps", 10)),
            base_loss_weight=float(cfg.get("base_loss_weight", 1.0)),
            consistency_loss_weight=float(cfg.get("consistency_loss_weight", 1.0)),
            stopgrad_target=bool(cfg.get("stopgrad_target", True)),
            step_sampling=str(cfg.get("step_sampling", "powers_of_two")),
            min_step=(None if cfg.get("min_step", None) is None else float(cfg.get("min_step"))),
            max_step=float(cfg.get("max_step", 0.5)),
            include_half_step=bool(cfg.get("include_half_step", True)),
            include_one_step_target=bool(cfg.get("include_one_step_target", True)),
            one_step_loss_weight=float(cfg.get("one_step_loss_weight", 1.0)),
        )
        if self.shortcut_cfg.num_base_steps <= 0:
            raise ValueError("shortcut.num_base_steps must be > 0")

        self.d_embed = nn.Sequential(
            nn.Linear(1, self.shortcut_cfg.embed_dim),
            nn.ReLU(),
            nn.Linear(self.shortcut_cfg.embed_dim, self.shortcut_cfg.embed_dim),
        )
        logger.info(
            "[ShortcutFlow] embed_dim=%d num_base_steps=%d base_w=%.3f cons_w=%.3f "
            "one_step_w=%.3f include_one_step=%s stopgrad=%s",
            self.shortcut_cfg.embed_dim,
            self.shortcut_cfg.num_base_steps,
            self.shortcut_cfg.base_loss_weight,
            self.shortcut_cfg.consistency_loss_weight,
            self.shortcut_cfg.one_step_loss_weight,
            self.shortcut_cfg.include_one_step_target,
            self.shortcut_cfg.stopgrad_target,
        )
        print(
            f"[ShortcutFlow] embed_dim={self.shortcut_cfg.embed_dim} "
            f"num_base_steps={self.shortcut_cfg.num_base_steps} "
            f"base_w={self.shortcut_cfg.base_loss_weight} cons_w={self.shortcut_cfg.consistency_loss_weight} "
            f"one_step_w={self.shortcut_cfg.one_step_loss_weight} "
            f"include_half_step={self.shortcut_cfg.include_half_step} "
            f"include_one_step_target={self.shortcut_cfg.include_one_step_target} "
            f"stopgrad_target={self.shortcut_cfg.stopgrad_target}"
        )

    def _augment_global_cond_with_d(self, nx: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        b = nx.shape[0]
        d_in = d.view(b, 1)
        d_emb = self.d_embed(d_in)
        return torch.cat([nx, d_emb], dim=-1)

    def _sample_shortcut_levels(self, device: torch.device) -> torch.Tensor:
        if self.shortcut_cfg.step_sampling != "powers_of_two":
            raise NotImplementedError(
                f"shortcut.step_sampling={self.shortcut_cfg.step_sampling} is not implemented"
            )
        k = float(self.shortcut_cfg.num_base_steps)
        d = 1.0 / k
        levels = []
        max_step = float(self.shortcut_cfg.max_step)
        min_step = self.shortcut_cfg.min_step
        while d <= 0.5 + 1e-12:
            if d <= max_step + 1e-12 and (min_step is None or d >= min_step - 1e-12):
                levels.append(d)
            d *= 2.0
        # Critical: include d=0.5 so consistency directly supervises big_step=1.0.
        if self.shortcut_cfg.include_half_step:
            levels.append(0.5)
        if not levels:
            base = 1.0 / k
            levels = [base]
        eps = 1e-12
        levels = [x for x in sorted(set(levels)) if x > 0.0 and (2.0 * x <= 1.0 + eps)]
        if self.shortcut_cfg.include_half_step:
            has_half = any(abs(x - 0.5) < 1e-6 for x in levels)
            assert has_half, "include_half_step=true requires d=0.5 in sampled levels"
        if not levels:
            raise ValueError("No valid shortcut d levels satisfy constraints after filtering.")
        logger.info("[ShortcutFlow] sampled shortcut levels=%s", levels)
        return torch.tensor(levels, device=device, dtype=torch.float32)

    def sample_shortcut_d_and_t(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        levels = self._sample_shortcut_levels(device)
        idx = torch.randint(low=0, high=levels.shape[0], size=(batch_size,), device=device)
        d = levels[idx].view(batch_size, 1, 1)
        t_max = torch.clamp(1.0 - 2.0 * d, min=0.0)
        t = torch.rand((batch_size, 1, 1), device=device) * t_max
        return d, t

    def _forward_shortcut(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        d: torch.Tensor,
        nx: torch.Tensor,
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

        z0 = self._init_noise(b)
        z1 = ny

        # Base shortcut loss with smallest step d_min.
        t_base = torch.rand((b, 1, 1), device=DEVICE)
        x_t = (1.0 - t_base) * z0 + t_base * z1
        v_target = z1 - z0
        d_min = torch.full((b, 1, 1), 1.0 / float(self.shortcut_cfg.num_base_steps), device=DEVICE)
        v_pred = self._forward_shortcut(x_t, t_base, d_min, nx, phase_flow=phase_flow)
        per_xyz, per_rot6d, per_grip = self._per_timestep_loss(v_pred, v_target)
        weights = self._compute_gripper_weights(ny)
        loss_base_xyz = (per_xyz * weights).mean()
        loss_base_rot6d = (per_rot6d * weights).mean()
        loss_base_grip = (per_grip * weights).mean()

        # Self-consistency: one big step ~= two small steps.
        d_sc, t_sc = self.sample_shortcut_d_and_t(b, DEVICE)
        x_sc = (1.0 - t_sc) * z0 + t_sc * z1
        v_big = self._forward_shortcut(x_sc, t_sc, 2.0 * d_sc, nx, phase_flow=phase_flow)
        x_big = x_sc + 2.0 * d_sc * v_big

        v1 = self._forward_shortcut(x_sc, t_sc, d_sc, nx, phase_flow=phase_flow)
        x_mid = x_sc + d_sc * v1
        v2 = self._forward_shortcut(x_mid, t_sc + d_sc, d_sc, nx, phase_flow=phase_flow)
        x_two = x_mid + d_sc * v2

        x_two_target = x_two.detach() if self.shortcut_cfg.stopgrad_target else x_two
        loss_sc = F.mse_loss(x_big, x_two_target)
        one_step_anchor_t_mean = 0.0
        loss_one_step = torch.zeros((), device=DEVICE, dtype=x_big.dtype)
        if self.shortcut_cfg.include_one_step_target:
            # One-step consistency is anchored on interpolation states, not only pure noise.
            # t_anchor is sampled in [0, 0.5] so (t_anchor + 0.5) stays in [0, 1].
            t_anchor = torch.rand((b, 1, 1), device=DEVICE) * 0.5
            x_anchor = (1.0 - t_anchor) * z0 + t_anchor * z1
            d_full = torch.ones((b, 1, 1), device=DEVICE)
            v_full = self._forward_shortcut(x_anchor, t_anchor, d_full, nx, phase_flow=phase_flow)
            x_full = x_anchor + d_full * v_full

            d_half = torch.full((b, 1, 1), 0.5, device=DEVICE)
            v_h1 = self._forward_shortcut(x_anchor, t_anchor, d_half, nx, phase_flow=phase_flow)
            x_half = x_anchor + d_half * v_h1
            t_half1 = t_anchor + 0.5
            v_h2 = self._forward_shortcut(x_half, t_half1, d_half, nx, phase_flow=phase_flow)
            x_half_half = x_half + d_half * v_h2
            x_half_half_target = x_half_half.detach() if self.shortcut_cfg.stopgrad_target else x_half_half
            loss_one_step = F.mse_loss(x_full, x_half_half_target)
            one_step_anchor_t_mean = float(t_anchor.mean().item())

        contains_half = bool(torch.any(torch.isclose(d_sc.view(-1), torch.tensor(0.5, device=DEVICE))).item())
        stats = {
            "shortcut/d_mean": float(d_sc.mean().item()),
            "shortcut/t_mean": float(t_sc.mean().item()),
            "shortcut/max_sampled_d": float(d_sc.max().item()),
            "shortcut/contains_half_step": 1.0 if contains_half else 0.0,
            "shortcut/include_one_step_target": 1.0 if self.shortcut_cfg.include_one_step_target else 0.0,
            "shortcut/one_step_anchor_t_mean": one_step_anchor_t_mean,
            "shortcut/num_base_steps": float(self.shortcut_cfg.num_base_steps),
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
        nx = self.encode_obs(pcd, robot_state_obs)
        b = nx.shape[0]
        z = self._init_noise(b) if noise is None else noise
        traj = [z]

        phase_seq = None
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

        k = int(self.num_k_infer)
        if k <= 0:
            raise ValueError("num_k_infer must be >= 1")

        if k == 1:
            d = torch.ones((b, 1, 1), device=DEVICE)
            t = torch.zeros((b, 1, 1), device=DEVICE)
            v = self._forward_shortcut(z, t, d, nx, phase_flow=phase_seq)
            z = z + d * v
            traj.append(z)
        else:
            d = torch.full((b, 1, 1), 1.0 / float(k), device=DEVICE)
            for i in range(k):
                t = torch.full((b, 1, 1), float(i) / float(k), device=DEVICE)
                v = self._forward_shortcut(z, t, d, nx, phase_flow=phase_seq)
                z = z + d * v
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
