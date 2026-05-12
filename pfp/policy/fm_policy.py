from __future__ import annotations
import functools
import logging
import os
import sys
import hydra
import torch
import torch.nn as nn
import pypose as pp
from omegaconf import DictConfig, OmegaConf
from composer.models import ComposerModel
from pfp.policy.base_policy import BasePolicy
from pfp import DEVICE, REPO_DIRS
from pfp.common.se3_utils import init_random_traj_th
from pfp.common.fm_utils import get_timesteps
from pfp.data.dataset_pcd import augment_pcd_data
from pfp.common.phase_utils import (
    PhaseConditioningConfig,
    compute_phase_labels_torch_from_gripper,
    phase_cfg_from,
)

logger = logging.getLogger(__name__)


def _unet_stem_in_channels(unet: nn.Module) -> int | None:
    """Best-effort read of ConditionalUnet1D first stem Conv1d in_channels (matches input_dim)."""
    try:
        conv = unet.down_modules[0][0].blocks[0].block[0]
        if isinstance(conv, nn.Conv1d):
            return int(conv.in_channels)
    except Exception:
        return None
    return None


def _unet_final_out_channels(unet: nn.Module) -> int | None:
    """ConditionalUnet1D final Conv1d out_channels (full vector field width; equals input_dim for vanilla net)."""
    try:
        fc = unet.final_conv[-1]
        if isinstance(fc, nn.Conv1d):
            return int(fc.out_channels)
    except Exception:
        return None
    return None


def _instantiate_diffusion_drop_legacy_output_dim(
    diffusion_net: functools.partial | DictConfig | dict,
    *,
    input_dim: int,
) -> nn.Module:
    """
    Vanilla diffusion_policy ConditionalUnet1D matches output width to input width when output_dim
    is omitted. Strip any legacy ``output_dim`` from Hydra partial/config so we always get a
    full [D+E]-channel field and slice the first D channels for velocity in FMPolicy.
    """
    if isinstance(diffusion_net, functools.partial):
        kw = dict(diffusion_net.keywords)
        kw.pop("output_dim", None)
        kw["input_dim"] = input_dim
        return diffusion_net.func(*diffusion_net.args, **kw)

    if isinstance(diffusion_net, DictConfig) or isinstance(diffusion_net, dict):
        container = OmegaConf.to_container(diffusion_net, resolve=True)
        if isinstance(container, dict):
            container.pop("output_dim", None)
            # Nested configs often keep ``_partial_: true``; completing with input_dim must yield a
            # full module, not another partial.
            if container.get("_partial_") is True:
                container["_partial_"] = False
            cfg = OmegaConf.create(container)
            return hydra.utils.instantiate(cfg, input_dim=input_dim)

    raise TypeError(f"expected partial or DictConfig/dict; got {type(diffusion_net)}")


def instantiate_diffusion_net_for_fm(
    diffusion_net: nn.Module | functools.partial | DictConfig | dict,
    *,
    input_dim: int,
) -> tuple[nn.Module, str]:
    """
    Build ConditionalUnet1D for FMPolicy.

    Only ``input_dim`` is passed through to the UNet. Standard ConditionalUnet1D uses the same
    channel width for input and output; phase conditioning uses ``input_dim = D + E`` and the
    extra output channels are discarded — only ``[..., :D]`` is used as the action velocity.

    Hydra with ``_partial_: true`` passes a ``functools.partial``; completing it yields a real
    ``nn.Module`` (never leave a bare partial as ``self.diffusion_net``).
    """
    if isinstance(diffusion_net, nn.Module):
        return diffusion_net, "already nn.Module"

    if isinstance(diffusion_net, functools.partial):
        m = _instantiate_diffusion_drop_legacy_output_dim(diffusion_net, input_dim=input_dim)
        if not isinstance(m, nn.Module):
            raise TypeError(f"partial did not return nn.Module: {type(m)}")
        return m, "functools.partial(...) completed (output_dim stripped if present)"

    if isinstance(diffusion_net, DictConfig) or isinstance(diffusion_net, dict):
        m = _instantiate_diffusion_drop_legacy_output_dim(diffusion_net, input_dim=input_dim)
        if not isinstance(m, nn.Module):
            raise TypeError(f"instantiated diffusion_net is not nn.Module: {type(m)}")
        return m, "hydra.utils.instantiate(DictConfig|dict, output_dim stripped if present)"

    if callable(diffusion_net):
        m = diffusion_net(input_dim=input_dim)
        if not isinstance(m, nn.Module):
            raise TypeError(f"callable did not return nn.Module: {type(m)}")
        return m, "callable constructor"

    raise TypeError(
        f"diffusion_net must be nn.Module, functools.partial, DictConfig, dict, or callable; "
        f"got {type(diffusion_net)}"
    )


def _warn_checkpoint_shape_mismatches(module: nn.Module, state_dict: dict, *, tag: str) -> None:
    """Before load_state_dict: warn if overlapping tensors differ in shape (e.g. phase_embed_dim)."""
    for key in ("phase_embedding.weight", "phase_embedding.bias"):
        if key not in state_dict:
            continue
        if key not in module.state_dict():
            continue
        t_ckpt = state_dict[key]
        t_mod = module.state_dict()[key]
        if t_ckpt.shape != t_mod.shape:
            print(
                f"[checkpoint:{tag}] WARNING: tensor shape mismatch for '{key}': "
                f"checkpoint {tuple(t_ckpt.shape)} vs model {tuple(t_mod.shape)} "
                f"(e.g. different phase_embed_dim). Load may fail or require strict=False partial load.",
                file=sys.stderr,
            )


def log_state_dict_load(
    module: nn.Module,
    state_dict: dict,
    *,
    strict: bool = False,
    tag: str = "load",
    phase_enabled: bool | None = None,
) -> tuple[list[str], list[str]]:
    """load_state_dict(strict=False) with stderr diagnostics for silent-compat loads."""
    _warn_checkpoint_shape_mismatches(module, state_dict, tag=tag)
    incompatible = module.load_state_dict(state_dict, strict=strict)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)

    def _emit(title: str, keys: list[str]) -> None:
        if not keys:
            return
        print(f"[checkpoint:{tag}] WARNING: {title} ({len(keys)} keys):", file=sys.stderr)
        show = keys[:80]
        for k in show:
            print(f"  - {k}", file=sys.stderr)
        if len(keys) > len(show):
            print(f"  ... and {len(keys) - len(show)} more", file=sys.stderr)

    _emit("missing_keys", missing)
    _emit("unexpected_keys", unexpected)

    if phase_enabled is not None:
        miss_phase_embed = any(k.startswith("phase_embedding.") for k in missing)
        unexp_phase_embed = any(k.startswith("phase_embedding.") for k in unexpected)
        miss_diffusion = any(k.startswith("diffusion_net.") for k in missing)
        if phase_enabled and miss_phase_embed:
            print(
                f"[checkpoint:{tag}] WARNING: loading into phase-conditioned model but checkpoint "
                f"has no phase_embedding weights (likely baseline checkpoint). Phase embedding stays randomly initialized.",
                file=sys.stderr,
            )
        if not phase_enabled and unexp_phase_embed:
            print(
                f"[checkpoint:{tag}] WARNING: checkpoint contains phase_embedding weights but "
                f"current model has phase conditioning disabled; unexpected phase keys ignored.",
                file=sys.stderr,
            )
        if phase_enabled and miss_diffusion:
            print(
                f"[checkpoint:{tag}] WARNING: diffusion_net keys missing — often baseline→phase "
                f"with mismatched UNet channel widths; verify training/inference use compatible checkpoints.",
                file=sys.stderr,
            )

    return missing, unexpected


class FMPolicy(ComposerModel, BasePolicy):
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        n_obs_steps: int,
        n_pred_steps: int,
        num_k_infer: int,
        time_conditioning: bool,
        obs_encoder: nn.Module,
        diffusion_net: nn.Module | functools.partial | DictConfig | dict,
        augment_data: bool = False,
        loss_weights: dict[int] = None,
        pos_emb_scale: int = 20,
        norm_pcd_center: list = None,
        noise_type: str = "gaussian",
        noise_scale: float = 1.0,
        loss_type: str = "l2",
        use_gripper_motion_weights: bool = False,
        gripper_motion_lambda: float = 3.0,
        gripper_motion_window: int = 1,
        flow_schedule: str = "linear",
        exp_scale: float = None,
        snr_sampler: str = "uniform",
        subs_factor: int = 1,
        phase_conditioning: dict | None = None,
    ) -> None:
        ComposerModel.__init__(self)
        BasePolicy.__init__(self, n_obs_steps, subs_factor)
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.n_obs_steps = n_obs_steps
        self.n_pred_steps = n_pred_steps
        self.pos_emb_scale = pos_emb_scale
        self.num_k_infer = num_k_infer
        self.time_conditioning = time_conditioning
        self.obs_encoder = obs_encoder
        self.norm_pcd_center = norm_pcd_center
        self.augment_data = augment_data
        self.noise_type = noise_type
        self.noise_scale = noise_scale
        self.ny_shape = (n_pred_steps, y_dim)
        self.l_w = loss_weights
        self.loss_type = loss_type
        self.use_gripper_motion_weights = use_gripper_motion_weights
        self.gripper_motion_lambda = gripper_motion_lambda
        self.gripper_motion_window = gripper_motion_window
        self.flow_schedule = flow_schedule
        self.exp_scale = exp_scale
        self.snr_sampler = snr_sampler
        self.phase_cfg: PhaseConditioningConfig = phase_cfg_from(phase_conditioning)
        self.phase_enabled = bool(self.phase_cfg.enabled)
        if self.phase_enabled:
            self.phase_embedding = nn.Embedding(self.phase_cfg.num_phases, self.phase_cfg.phase_embed_dim)
        else:
            self.phase_embedding = None
        effective_input_dim = int(y_dim)
        if self.phase_enabled:
            effective_input_dim += int(self.phase_cfg.phase_embed_dim)

        self.diffusion_net, ctor_src = instantiate_diffusion_net_for_fm(
            diffusion_net,
            input_dim=effective_input_dim,
        )
        self._diffusion_net_ctor_source = ctor_src
        self.diffusion_net_input_dim = effective_input_dim

        assert isinstance(self.diffusion_net, nn.Module), "diffusion_net must be nn.Module after init"
        stem_in = _unet_stem_in_channels(self.diffusion_net)
        if stem_in is not None and stem_in != effective_input_dim:
            raise AssertionError(
                f"UNet stem in_channels={stem_in} != effective_input_dim={effective_input_dim}"
            )
        final_out = _unet_final_out_channels(self.diffusion_net)
        if final_out is not None and final_out != effective_input_dim:
            raise AssertionError(
                f"UNet final out_channels={final_out} != effective_input_dim={effective_input_dim} "
                f"(vanilla ConditionalUnet1D uses same width for input and full output field)"
            )

        print(
            f"[FMPolicy] diffusion_net: {ctor_src}\n"
            f"  phase_conditioning.enabled={self.phase_enabled}  y_dim={y_dim}  "
            f"phase_embed_dim={int(self.phase_cfg.phase_embed_dim) if self.phase_enabled else 0}\n"
            f"  UNet full field width (in/out)={effective_input_dim}  "
            f"action velocity uses first {int(y_dim)} channels only",
            file=sys.stderr,
        )
        logger.info(
            "FMPolicy diffusion_net ctor=%s phase_enabled=%s y_dim=%s effective_input_dim=%s",
            ctor_src,
            self.phase_enabled,
            y_dim,
            effective_input_dim,
        )

        if loss_type == "l2":
            self.loss_fun = nn.MSELoss()
        elif loss_type == "l1":
            self.loss_fun = nn.L1Loss()
        else:
            raise NotImplementedError
        return

    def set_num_k_infer(self, num_k_infer: int):
        self.num_k_infer = num_k_infer
        return

    def _slice_velocity(self, vel_full: torch.Tensor) -> torch.Tensor:
        """
        Vanilla ConditionalUnet1D outputs the same channel width as ``input_dim`` (D or D+E).
        Only the first D channels are physical action velocity; any extra channels are discarded.
        """
        D = int(self.y_dim)
        io = int(self.diffusion_net_input_dim)
        assert vel_full.shape[-1] == io, (
            f"UNet output width {vel_full.shape[-1]} != diffusion_net_input_dim={io}"
        )
        assert vel_full.shape[-1] >= D
        vel = vel_full[..., :D]
        assert vel.shape[-1] == D
        return vel

    def set_flow_schedule(self, flow_schedule: str, exp_scale: float):
        self.flow_schedule = flow_schedule
        self.exp_scale = exp_scale
        return

    def _norm_obs(self, pcd: torch.Tensor) -> torch.Tensor:
        # I only do centering here, no scaling, to keep the relative distances and interpretability
        pcd[..., :3] -= torch.tensor(self.norm_pcd_center, device=DEVICE)
        return pcd

    def _norm_robot_state(self, robot_state: torch.Tensor) -> torch.Tensor:
        # I only do centering here, no scaling, to keep the relative distances and interpretability
        robot_state[..., :3] -= torch.tensor(self.norm_pcd_center, device=DEVICE)
        robot_state[..., 9] -= torch.tensor(0.5, device=DEVICE)
        return robot_state

    def _denorm_robot_state(self, robot_state: torch.Tensor) -> torch.Tensor:
        robot_state[..., :3] += torch.tensor(self.norm_pcd_center, device=DEVICE)
        robot_state[..., 9] += torch.tensor(0.5, device=DEVICE)
        return robot_state

    @staticmethod
    def _denorm_gripper(gripper_norm: torch.Tensor) -> torch.Tensor:
        """
        Undo _norm_robot_state gripper shift (raw open≈1, closed≈0 → norm subtracts 0.5).
        Phase heuristics in phase_utils use raw [0,1] scale and gripper_close_threshold.
        """
        return gripper_norm + 0.5

    def _norm_data(self, batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        # batch may be (pcd, robot_state_obs, robot_state_pred) or (..., phase_pred)
        pcd, robot_state_obs, robot_state_pred = batch[:3]
        pcd = self._norm_obs(pcd)
        robot_state_obs = self._norm_robot_state(robot_state_obs)
        robot_state_pred = self._norm_robot_state(robot_state_pred)
        if len(batch) >= 4:
            return pcd, robot_state_obs, robot_state_pred, batch[3]
        return pcd, robot_state_obs, robot_state_pred

    def _augment_data(self, batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        return augment_pcd_data(batch)

    def _init_noise(self, batch_size: int) -> torch.Tensor:
        B = batch_size
        T = self.n_pred_steps
        if self.noise_type == "gaussian":
            noise = torch.randn((batch_size, *self.ny_shape), device=DEVICE)
            return noise * self.noise_scale
        elif self.noise_type == "trajectory":
            return init_random_traj_th(batch_size, self.n_pred_steps, self.noise_scale)
        elif self.noise_type == "igso3":
            noise_pos = torch.randn((B, T, 3), device=DEVICE)
            noise_rot = pp.randn_SO3((B, T), device=DEVICE).matrix()
            noise_gripper = torch.randn((B, T, 1), device=DEVICE)
            noise = torch.cat(
                [noise_pos, noise_rot[..., :3, 0], noise_rot[..., :3, 1], noise_gripper], dim=-1
            )
            return noise
        else:
            raise NotImplementedError

    def _sample_snr(self, batch_size: int) -> torch.Tensor:
        if self.snr_sampler == "uniform":
            return torch.rand((batch_size, 1, 1), device=DEVICE)
        elif self.snr_sampler == "logit_normal":
            return torch.sigmoid(torch.randn((batch_size, 1, 1), device=DEVICE))
        else:
            raise NotImplementedError

    def _per_timestep_loss(
        self, pred_vel: torch.Tensor, target_vel: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return per-timestep losses for xyz, rot6d, grip with shape [B, T]."""
        if self.loss_type == "l2":
            diff_xyz = pred_vel[..., :3] - target_vel[..., :3]
            diff_rot6d = pred_vel[..., 3:9] - target_vel[..., 3:9]
            diff_grip = pred_vel[..., 9] - target_vel[..., 9]
            per_xyz = (diff_xyz ** 2).mean(dim=-1)
            per_rot6d = (diff_rot6d ** 2).mean(dim=-1)
            per_grip = (diff_grip ** 2)
        elif self.loss_type == "l1":
            diff_xyz = pred_vel[..., :3] - target_vel[..., :3]
            diff_rot6d = pred_vel[..., 3:9] - target_vel[..., 3:9]
            diff_grip = pred_vel[..., 9] - target_vel[..., 9]
            per_xyz = diff_xyz.abs().mean(dim=-1)
            per_rot6d = diff_rot6d.abs().mean(dim=-1)
            per_grip = diff_grip.abs()
        else:
            raise NotImplementedError
        return per_xyz, per_rot6d, per_grip

    def _compute_gripper_weights(self, ny: torch.Tensor) -> torch.Tensor:
        """Return time weights [B, T] around gripper state changes, normalized to mean 1."""
        B, T, _ = ny.shape
        g = ny[..., 9]
        if not self.use_gripper_motion_weights or T <= 1:
            return torch.ones((B, T), device=ny.device, dtype=ny.dtype)

        dg = g[:, 1:] - g[:, :-1]
        transitions = (dg.abs() > 0.5)
        mark = torch.zeros_like(g, dtype=torch.bool)
        mark[:, 1:] = transitions

        important = mark.clone()
        w = self.gripper_motion_window
        for k in range(1, w + 1):
            important |= mark.roll(shifts=k, dims=1)
            important |= mark.roll(shifts=-k, dims=1)

        weights = torch.ones_like(g, dtype=ny.dtype)
        weights[important] = self.gripper_motion_lambda
        weights = weights / (weights.mean(dim=1, keepdim=True) + 1e-8)
        return weights

    # ############### Training ################

    def forward(self, batch):
        """batch is the output of the dataloader"""
        return 0

    def loss(self, outputs, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        outputs: the output of the forward pass
        batch: the output of the dataloader
        """
        with torch.no_grad():
            batch = self._norm_data(batch)
            if self.augment_data:
                batch = self._augment_data(batch)
        pcd, robot_state_obs, robot_state_pred = batch[:3]
        phase_pred = batch[3] if (self.phase_enabled and len(batch) >= 4) else None
        if self.phase_enabled and phase_pred is None:
            raise ValueError(
                "phase_conditioning.enabled=true, but dataset batch has no phase tensor. "
                "Pass phase_conditioning to dataset and ensure it returns phase_pred."
            )
        loss_xyz, loss_rot6d, loss_grip = self.calculate_loss(
            pcd, robot_state_obs, robot_state_pred, phase=phase_pred
        )
        loss = (
            self.l_w["xyz"] * loss_xyz
            + self.l_w["rot6d"] * loss_rot6d
            + self.l_w["grip"] * loss_grip
        )
        self.logger.log_metrics(
            {
                "loss/train/xyz": loss_xyz.item(),
                "loss/train/rot6d": loss_rot6d.item(),
                "loss/train/grip": loss_grip.item(),
            }
        )
        return loss

    def calculate_loss(
        self,
        pcd: torch.Tensor,
        robot_state_obs: torch.Tensor,
        robot_state_pred: torch.Tensor,
        *,
        phase: torch.Tensor | None = None,
    ):
        nx: torch.Tensor = self.obs_encoder(pcd, robot_state_obs)
        ny: torch.Tensor = robot_state_pred
        B, T, _ = ny.shape
        t = self._sample_snr(B)
        z0 = self._init_noise(ny.shape[0])
        z1 = ny
        z_flow = t * z1 + (1.0 - t) * z0
        D = int(self.y_dim)
        assert z_flow.shape[-1] == D
        if self.phase_enabled:
            if phase is None:
                raise ValueError("phase_conditioning.enabled=true requires phase labels in calculate_loss().")
            if phase.dtype != torch.int64:
                phase = phase.to(torch.int64)
            if phase.shape != (B, T):
                raise ValueError(f"phase must be (B,T)={B,T}, got {tuple(phase.shape)}")
            if torch.any((phase < 0) | (phase >= self.phase_cfg.num_phases)):
                raise ValueError("phase has values outside [0, num_phases-1].")
            phase_emb = self.phase_embedding(phase)  # (B,T,E)
            z_t = torch.cat([z_flow, phase_emb], dim=-1)  # (B,T,D+E)
            assert z_t.shape[-1] == D + int(self.phase_cfg.phase_embed_dim)
        else:
            z_t = z_flow
        target_vel = z1 - z0
        timesteps = t.squeeze() * self.pos_emb_scale if self.time_conditioning else None
        pred_vel_full = self.diffusion_net(z_t, timesteps, global_cond=nx)
        pred_vel = self._slice_velocity(pred_vel_full)
        per_xyz, per_rot6d, per_grip = self._per_timestep_loss(pred_vel, target_vel)
        weights = self._compute_gripper_weights(ny)

        loss_xyz = (per_xyz * weights).mean()
        loss_rot6d = (per_rot6d * weights).mean()
        loss_grip = (per_grip * weights).mean()
        return loss_xyz, loss_rot6d, loss_grip

    # ############### Inference ################

    def eval_forward(self, batch: tuple[torch.Tensor, ...], outputs=None) -> torch.Tensor:
        """
        batch: the output of the eval dataloader
        outputs: the output of the forward pass
        """
        batch = self._norm_data(batch)
        pcd, robot_state_obs, robot_state_pred = batch[:3]
        phase_pred = batch[3] if (self.phase_enabled and len(batch) >= 4) else None

        # Eval loss
        loss_xyz, loss_rot6d, loss_grip = self.calculate_loss(
            pcd, robot_state_obs, robot_state_pred, phase=phase_pred
        )
        loss_total = (
            self.l_w["xyz"] * loss_xyz
            + self.l_w["rot6d"] * loss_rot6d
            + self.l_w["grip"] * loss_grip
        )
        self.logger.log_metrics(
            {
                "loss/eval/xyz": loss_xyz.item(),
                "loss/eval/rot6d": loss_rot6d.item(),
                "loss/eval/grip": loss_grip.item(),
                "loss/eval/total": loss_total.item(),
            }
        )

        # Eval metrics
        # Offline eval: if phase labels exist in the dataset, use them consistently.
        pred_y = self.infer_y(pcd, robot_state_obs, phase=phase_pred)
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
        nx: torch.Tensor = self.obs_encoder(pcd, robot_state_obs)
        B = nx.shape[0]
        z = self._init_noise(B) if noise is None else noise
        traj = [z]
        phase_seq = None
        g_last_norm: torch.Tensor | None = None
        g_last_raw: torch.Tensor | None = None
        if self.phase_enabled:
            # If phase not provided, use heuristic over horizon based on current gripper state.
            if phase is None:
                # robot_state_obs is normalized before infer_y (see infer_from_np / _norm_robot_state).
                # Phase thresholds are defined in raw gripper scale, so convert back.
                # robot_state_obs: (B, n_obs, 10)
                g_last_norm = robot_state_obs[:, -1, 9]
                g_last_raw = self._denorm_gripper(g_last_norm)
                ph, _ = compute_phase_labels_torch_from_gripper(
                    g_last_raw.view(B, 1).repeat(1, self.n_pred_steps),
                    thr=self.phase_cfg.gripper_close_threshold,
                    contact_window=self.phase_cfg.contact_window,
                    num_phases=self.phase_cfg.num_phases,
                )
                # If gripper is open, we want phase 0 early, then contact window, then phase 2.
                # Approximate grasp time at middle of horizon.
                open_mask = (g_last_raw >= self.phase_cfg.gripper_close_threshold).view(B, 1)
                if bool(open_mask.any()):
                    T = self.n_pred_steps
                    t_grasp = max(0, min(T - 1, int(0.5 * T)))
                    w = max(0, int(self.phase_cfg.contact_window))
                    tt = torch.arange(T, device=DEVICE).view(1, T)
                    base = torch.where(tt <= (t_grasp + w), torch.zeros_like(tt), torch.full_like(tt, 2))
                    win = (tt >= (t_grasp - w)) & (tt <= (t_grasp + w))
                    base = torch.where(win, torch.ones_like(base), base).to(torch.int64)
                    ph = torch.where(open_mask, base.expand(B, T), ph)
                phase_seq = ph
            else:
                if phase.dtype != torch.int64:
                    phase = phase.to(torch.int64)
                if phase.ndim == 1:
                    phase_seq = phase.view(B, 1).repeat(1, self.n_pred_steps)
                else:
                    phase_seq = phase
            if phase_seq.shape != (B, self.n_pred_steps):
                raise ValueError(f"phase must broadcast to (B,T)={(B,self.n_pred_steps)}, got {tuple(phase_seq.shape)}")
            assert phase_seq.dtype == torch.int64
            assert torch.all((phase_seq >= 0) & (phase_seq < self.phase_cfg.num_phases))

            thr = float(self.phase_cfg.gripper_close_threshold)
            if g_last_raw is not None and os.environ.get("PFP_PHASE_INFER_DEBUG", "").lower() in ("1", "true", "yes"):
                flat = phase_seq.reshape(-1)
                counts = {int(p): int((flat == p).sum().item()) for p in range(self.phase_cfg.num_phases)}
                logger.info(
                    "[phase-infer] g_norm min/max/mean=%.4f/%.4f/%.4f  g_raw min/max/mean=%.4f/%.4f/%.4f  thr=%.4f  counts=%s",
                    float(g_last_norm.min()),
                    float(g_last_norm.max()),
                    float(g_last_norm.float().mean()),
                    float(g_last_raw.min()),
                    float(g_last_raw.max()),
                    float(g_last_raw.float().mean()),
                    thr,
                    counts,
                )
            elif g_last_raw is not None and logger.isEnabledFor(logging.DEBUG):
                flat = phase_seq.reshape(-1)
                counts = {int(p): int((flat == p).sum().item()) for p in range(self.phase_cfg.num_phases)}
                logger.debug(
                    "[phase-infer] g_norm min/max/mean=%.4f/%.4f/%.4f  g_raw min/max/mean=%.4f/%.4f/%.4f  thr=%.4f  counts=%s",
                    float(g_last_norm.min()),
                    float(g_last_norm.max()),
                    float(g_last_norm.float().mean()),
                    float(g_last_raw.min()),
                    float(g_last_raw.max()),
                    float(g_last_raw.float().mean()),
                    thr,
                    counts,
                )

            if g_last_raw is not None:
                open_b = g_last_raw >= thr
                bad_first = open_b & (phase_seq[:, 0] == 2)
                if bool(bad_first.any()):
                    logger.warning(
                        "[phase-infer] Open gripper (raw >= %.3f) but phase_seq[:,0]==2 for %d/%d batch rows — "
                        "check phase heuristic / normalization.",
                        thr,
                        int(bad_first.sum().item()),
                        int(B),
                    )
        D = int(self.y_dim)
        E = int(self.phase_cfg.phase_embed_dim) if self.phase_enabled else 0
        t0, dt = get_timesteps(self.flow_schedule, self.num_k_infer, exp_scale=self.exp_scale)
        for i in range(self.num_k_infer):
            assert z.shape[-1] == D
            timesteps = torch.ones((B), device=DEVICE) * t0[i]
            timesteps *= self.pos_emb_scale
            if self.phase_enabled:
                phase_emb = self.phase_embedding(phase_seq)  # (B,T,E)
                z_in = torch.cat([z, phase_emb], dim=-1)  # (B,T,D+E)
                assert z_in.shape[-1] == D + E
            else:
                z_in = z
            vel_full = self.diffusion_net(z_in, timesteps, global_cond=nx)
            vel = self._slice_velocity(vel_full)
            z = z.detach().clone() + vel * dt[i]
            assert z.shape[-1] == D
            traj.append(z)

        if return_traj:
            return torch.stack(traj)
        return traj[-1]

    @classmethod
    def load_from_checkpoint(
        cls,
        ckpt_name: str,
        ckpt_episode: str,
        num_k_infer: int,
        flow_schedule: str = None,
        exp_scale: float = None,
        subs_factor: int = 1,
    ):
        ckpt_dir = REPO_DIRS.CKPT / ckpt_name
        ckpt_path_list = list(ckpt_dir.glob(f"{ckpt_episode}*"))
        assert len(ckpt_path_list) > 0, f"No checkpoint found in {ckpt_dir} with {ckpt_episode}"
        assert len(ckpt_path_list) < 2, f"Multiple ckpts found in {ckpt_dir} with {ckpt_episode}"
        ckpt_fpath = ckpt_path_list[0]

        # Чекпоинт мог быть сохранён с NumPy 2.x (numpy._core); при NumPy 1.x подменяем для unpickle
        import sys
        import numpy as _np
        if "numpy._core" not in sys.modules and hasattr(_np, "core"):
            sys.modules["numpy._core"] = _np.core

        state_dict = torch.load(ckpt_fpath, map_location=DEVICE, weights_only=False)
        cfg = OmegaConf.load(ckpt_dir / "config.yaml")
        # cfg.model.obs_encoder.encoder.random_crop = False
        cfg.model.subs_factor = subs_factor
        assert cfg.model._target_.split(".")[-1] == cls.__name__
        pcfg = getattr(cfg, "phase_conditioning", None)
        model: FMPolicy = hydra.utils.instantiate(cfg.model, phase_conditioning=pcfg)
        log_state_dict_load(
            model,
            state_dict["state"]["model"],
            strict=False,
            tag=f"{ckpt_name}/{ckpt_episode}",
            phase_enabled=bool(getattr(pcfg, "enabled", False)) if pcfg is not None else False,
        )
        model.to(DEVICE)
        # Ensure model is in float32 (model.to() should handle this, but ensure it)
        model = model.float()
        model.eval()
        if flow_schedule is not None:
            model.set_flow_schedule(flow_schedule, exp_scale)
        if num_k_infer is not None:
            model.set_num_k_infer(num_k_infer)
        return model


class FMPolicyImage(FMPolicy):

    def _norm_obs(self, image: torch.Tensor) -> torch.Tensor:
        """
        Image normalization is already done in the backbone, so here we just make it float
        """
        image = image.float() / 255.0
        return image

    def _augment_data(self, batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError
