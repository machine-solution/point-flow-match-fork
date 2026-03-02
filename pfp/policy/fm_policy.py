from __future__ import annotations
import hydra
import torch
import torch.nn as nn
import pypose as pp
from omegaconf import OmegaConf
from composer.models import ComposerModel
from pfp.policy.base_policy import BasePolicy
from pfp import DEVICE, REPO_DIRS
from pfp.common.se3_utils import init_random_traj_th
from pfp.common.fm_utils import get_timesteps
from pfp.data.dataset_pcd import augment_pcd_data


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
        diffusion_net: nn.Module,
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
        self.diffusion_net = diffusion_net
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

    def _norm_data(self, batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        pcd, robot_state_obs, robot_state_pred = batch
        pcd = self._norm_obs(pcd)
        robot_state_obs = self._norm_robot_state(robot_state_obs)
        robot_state_pred = self._norm_robot_state(robot_state_pred)
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
        pcd, robot_state_obs, robot_state_pred = batch
        loss_xyz, loss_rot6d, loss_grip = self.calculate_loss(
            pcd, robot_state_obs, robot_state_pred
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
        self, pcd: torch.Tensor, robot_state_obs: torch.Tensor, robot_state_pred: torch.Tensor
    ):
        nx: torch.Tensor = self.obs_encoder(pcd, robot_state_obs)
        ny: torch.Tensor = robot_state_pred
        B, T, _ = ny.shape
        t = self._sample_snr(B)
        z0 = self._init_noise(ny.shape[0])
        z1 = ny
        z_t = t * z1 + (1.0 - t) * z0
        target_vel = z1 - z0
        timesteps = t.squeeze() * self.pos_emb_scale if self.time_conditioning else None
        pred_vel = self.diffusion_net(z_t, timesteps, global_cond=nx)
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
        pcd, robot_state_obs, robot_state_pred = batch

        # Eval loss
        loss_xyz, loss_rot6d, loss_grip = self.calculate_loss(
            pcd, robot_state_obs, robot_state_pred
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
        pred_y = self.infer_y(pcd, robot_state_obs)
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
        noise=None,
        return_traj=False,
    ) -> torch.Tensor:
        nx: torch.Tensor = self.obs_encoder(pcd, robot_state_obs)
        B = nx.shape[0]
        z = self._init_noise(B) if noise is None else noise
        traj = [z]
        t0, dt = get_timesteps(self.flow_schedule, self.num_k_infer, exp_scale=self.exp_scale)
        for i in range(self.num_k_infer):
            timesteps = torch.ones((B), device=DEVICE) * t0[i]
            timesteps *= self.pos_emb_scale
            vel_pred = self.diffusion_net(z, timesteps, global_cond=nx)
            z = z.detach().clone() + vel_pred * dt[i]
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
        model: FMPolicy = hydra.utils.instantiate(cfg.model)
        model.load_state_dict(state_dict["state"]["model"], strict=False)
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
