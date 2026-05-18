import os
import shutil
import sys
from pathlib import Path
import hydra
import wandb
import subprocess
import torch
from typing import Any
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig
from omegaconf.base import ContainerMetadata
import torch.serialization as ts
from torch.utils.data import DataLoader
from composer.trainer import Trainer
from composer.loggers import WandBLogger
from composer.callbacks import LRMonitor
from composer.models import ComposerModel
from composer.algorithms import EMA
from composer.core import Callback, Event
try:
    # diffusion_policy uses diffusers.schedulers; some environments may have an incompatible
    # huggingface_hub/diffusers combo. We fall back to a simple torch scheduler for smoke runs.
    from diffusion_policy.model.common.lr_scheduler import get_scheduler  # type: ignore
except Exception as e:  # pragma: no cover
    print(f"[lr_scheduler] WARNING: failed to import diffusion_policy get_scheduler: {e}")
    print("[lr_scheduler] Falling back to torch.optim.lr_scheduler.LambdaLR (linear warmup then constant).")
    import torch

    def get_scheduler(name, *, optimizer, num_warmup_steps: int, num_training_steps: int):
        num_warmup_steps = int(num_warmup_steps)
        num_training_steps = int(num_training_steps)

        def lr_lambda(step: int) -> float:
            if num_warmup_steps <= 0:
                return 1.0
            if step < num_warmup_steps:
                return float(step) / float(max(1, num_warmup_steps))
            return 1.0

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
from pfp import DEVICE, DATA_DIRS, REPO_DIRS, set_seeds
from pfp.data.dataset_pcd import RobotDatasetPcd
from pfp.data.dataset_images import RobotDatasetImages
from pfp.policy.fm_policy import log_state_dict_load


if hasattr(ts, "add_safe_globals"):
    ts.add_safe_globals([ListConfig, ContainerMetadata, Any, list])

def _tensor_size_mb(t: torch.Tensor) -> float:
    return t.numel() * t.element_size() / (1024 ** 2)


def _log_gpu_memory(tag: str):
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated(0) / 1024 ** 2
    reserved = torch.cuda.memory_reserved(0) / 1024 ** 2
    max_alloc = torch.cuda.max_memory_allocated(0) / 1024 ** 2
    print(f"[memory] {tag}: allocated={alloc:.1f} MB  reserved={reserved:.1f} MB  max_allocated={max_alloc:.1f} MB")


class MemoryProfileCallback(Callback):
    """Принты по памяти внутри цикла обучения (INIT → FIT_START → первый батч)."""

    def run_event(self, event: Event, state, logger):
        if not torch.cuda.is_available():
            return
        batch = getattr(state.timestamp, "batch", 0) if state.timestamp else 0
        if event == Event.INIT:
            print("[memory] callback: INIT (model on device)")
            _log_gpu_memory("  INIT")
        elif event == Event.FIT_START:
            print("[memory] callback: FIT_START")
            _log_gpu_memory("  FIT_START")
        elif event == Event.BATCH_START and batch == 0:
            print("[memory] callback: BATCH_START batch=0")
            _log_gpu_memory("  BATCH_START 0")
        elif event == Event.AFTER_FORWARD and batch == 0:
            print("[memory] callback: AFTER_FORWARD batch=0")
            _log_gpu_memory("  AFTER_FORWARD 0")
        elif event == Event.AFTER_BACKWARD and batch == 0:
            print("[memory] callback: AFTER_BACKWARD batch=0")
            _log_gpu_memory("  AFTER_BACKWARD 0")


class ResourceMonitorCallback(Callback):
    """Раз в эпоху пишет в лог: диск (свободно), RAM/swap, GPU. Чтобы при падении было видно, что закончилось."""

    def __init__(self, path_to_check=None):
        self.path_to_check = path_to_check or os.getcwd()

    def run_event(self, event: Event, state, logger):
        if event != Event.EPOCH_END:
            return
        epoch = getattr(state.timestamp, "epoch", 0) if state.timestamp else 0
        lines = [f"[resources] Epoch {epoch} —"]

        # Диск
        try:
            du = shutil.disk_usage(self.path_to_check)
            free_gb = du.free / (1024 ** 3)
            total_gb = du.total / (1024 ** 3)
            lines.append(f"  disk: {free_gb:.1f} GB free / {total_gb:.1f} GB total ({self.path_to_check})")
        except Exception as e:
            lines.append(f"  disk: error {e}")

        # RAM и swap (через free -h, одна строка)
        try:
            r = subprocess.run(
                ["free", "-h"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if r.returncode == 0 and r.stdout:
                for line in r.stdout.strip().split("\n"):
                    if line.startswith("Mem:") or line.startswith("Swap:"):
                        lines.append(f"  {line.strip()}")
        except Exception as e:
            lines.append(f"  RAM/swap: error {e}")

        # GPU
        if torch.cuda.is_available():
            a = torch.cuda.memory_allocated(0) / 1024 ** 3
            r = torch.cuda.memory_reserved(0) / 1024 ** 3
            m = torch.cuda.max_memory_allocated(0) / 1024 ** 3
            lines.append(f"  GPU: alloc={a:.2f} GB  reserved={r:.2f} GB  max_alloc={m:.2f} GB")

        print("\n".join(lines))


class MilestoneCheckpointCopyCallback(Callback):
    """
    После сохранения Composer копирует файл эпохи N в milestone_ep{N}.pt
    (rolling save_num_checkpoints_to_keep их не трогает — это отдельные копии).
    """

    def __init__(self, milestones: list[int]):
        self.milestones = set(int(x) for x in milestones)

    def run_event(self, event: Event, state, logger):
        if event != Event.EPOCH_END:
            return
        epoch = int(state.timestamp.epoch)
        if epoch not in self.milestones:
            return
        run_name = getattr(state, "run_name", None)
        if not run_name:
            print(f"[milestone] skip: no state.run_name at epoch {epoch}")
            return
        ckpt_dir = REPO_DIRS.CKPT / str(run_name)
        if not ckpt_dir.is_dir():
            print(f"[milestone] skip: missing dir {ckpt_dir}")
            return
        matches = sorted(ckpt_dir.glob(f"ep{epoch:05d}*.pt"))
        if not matches:
            matches = sorted(ckpt_dir.glob(f"ep{epoch}*.pt"))
        if not matches:
            print(f"[milestone] no checkpoint file for epoch {epoch} in {ckpt_dir}")
            return
        src = matches[-1]
        dst = ckpt_dir / f"milestone_ep{epoch}.pt"
        shutil.copy2(src, dst)
        print(f"[milestone] saved {dst} (from {src.name})")


def _resolve_dataset_path(cfg: OmegaConf, *, train: bool) -> Path:
    key = "dataset_path_train" if train else "dataset_path_valid"
    override = getattr(cfg, key, None)
    if override is not None:
        s = str(override).strip()
        if s and s.lower() not in ("null", "none", "~"):
            return Path(s).expanduser().resolve()
    sub = "train" if train else "valid"
    return (DATA_DIRS.PFP / cfg.task_name / sub).resolve()


def _log_memory_usage(cfg: OmegaConf, composer_model: ComposerModel, optimizer, dataset_train):
    """Log model size, optimizer/EMA footprint, and one sample size (no dataloader iteration)."""
    n_params = sum(p.numel() for p in composer_model.parameters())
    n_trainable = sum(p.numel() for p in composer_model.parameters() if p.requires_grad)
    model_mb = n_params * 4 / (1024 ** 2)  # fp32
    print("[memory] === Model ===")
    print(f"[memory] Parameters: {n_params:,} (trainable {n_trainable:,})")
    print(f"[memory] Model weights (fp32): {model_mb:.2f} MB")

    # AdamW: 2 states (exp_avg, exp_avg_sq) per param, same dtype as param
    optimizer_mb = 2 * n_trainable * 4 / (1024 ** 2)
    print(f"[memory] Optimizer states (AdamW, fp32): ~{optimizer_mb:.2f} MB")

    if cfg.use_ema:
        ema_mb = n_params * 4 / (1024 ** 2)
        print(f"[memory] EMA copy (fp32): ~{ema_mb:.2f} MB")

    total_static_mb = model_mb + optimizer_mb + (model_mb if cfg.use_ema else 0)
    print(f"[memory] Total static (model + optimizer + EMA): ~{total_static_mb:.2f} MB")

    if torch.cuda.is_available():
        total_gpu = torch.cuda.get_device_properties(0).total_memory
        allocated = torch.cuda.memory_allocated(0)
        reserved = torch.cuda.memory_reserved(0)
        print("[memory] === GPU (before Trainer moves model) ===")
        print(f"[memory] Device total: {total_gpu / 1024**3:.2f} GB")
        print(f"[memory] Allocated: {allocated / 1024**2:.2f} MB")
        print(f"[memory] Reserved: {reserved / 1024**2:.2f} MB")

    # Размер одного сэмпла из датасета (не итерируем dataloader_train — иначе Composer ругается на active iterator)
    sample = dataset_train[0]
    if isinstance(sample, (list, tuple)):
        print("[memory] === One train sample (tensors) ===")
        for i, t in enumerate(sample):
            if isinstance(t, torch.Tensor):
                print(f"[memory]   sample[{i}] shape={tuple(t.shape)} dtype={t.dtype} -> {_tensor_size_mb(t):.2f} MB")
        sample_mb = sum(_tensor_size_mb(t) for t in sample if isinstance(t, torch.Tensor))
    else:
        sample_mb = _tensor_size_mb(sample)
        print(f"[memory] One train sample: {sample_mb:.2f} MB")
    batch_size = cfg.dataloader.get("batch_size", 128)
    print(f"[memory] Estimated batch (×{batch_size}) ~{sample_mb * batch_size:.2f} MB on GPU + activations")
    print("[memory] ===")


@hydra.main(version_base=None, config_path="../conf", config_name="train")
def main(cfg: OmegaConf):
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg))
    set_seeds(cfg.seed)

    use_val = bool(getattr(cfg, "use_validation", True))
    data_path_train = _resolve_dataset_path(cfg, train=True)
    print(f"[data] train: {data_path_train}")
    if use_val:
        data_path_valid = _resolve_dataset_path(cfg, train=False)
        print(f"[data] valid: {data_path_valid}")
    else:
        data_path_valid = None
        print("[data] valid: (disabled, use_validation=false)")

    if cfg.obs_mode == "pcd":
        dataset_train = RobotDatasetPcd(data_path_train, phase_conditioning=getattr(cfg, "phase_conditioning", None), **cfg.dataset)
        dataset_valid = (
            RobotDatasetPcd(data_path_valid, phase_conditioning=getattr(cfg, "phase_conditioning", None), **cfg.dataset) if use_val else None
        )
    elif cfg.obs_mode == "rgb":
        dataset_train = RobotDatasetImages(data_path_train, phase_conditioning=getattr(cfg, "phase_conditioning", None), **cfg.dataset)
        dataset_valid = (
            RobotDatasetImages(data_path_valid, phase_conditioning=getattr(cfg, "phase_conditioning", None), **cfg.dataset) if use_val else None
        )
    else:
        raise ValueError(f"Unknown observation mode: {cfg.obs_mode}")

    # Phase conditioning logging (optional)
    pcfg = getattr(cfg, "phase_conditioning", None)
    if pcfg is not None and bool(getattr(pcfg, "enabled", False)):
        try:
            stats = dataset_train.phase_stats() if hasattr(dataset_train, "phase_stats") else None
            if stats:
                print(f"[phase] enabled=True  num_phases={stats.get('num_phases')}  embed_dim={getattr(pcfg, 'phase_embed_dim', None)}")
                print(f"[phase] contact_window={stats.get('contact_window')} thr={stats.get('gripper_close_threshold')}")
                print(f"[phase] episodes_scanned={stats.get('episodes_scanned')} no_grasp={stats.get('no_grasp_episodes')}")
                print(f"[phase] phase_fracs={stats.get('phase_fracs')} avg_grasp_timestep={stats.get('avg_grasp_timestep')}")
        except Exception as e:
            print(f"[phase] warning: could not compute dataset phase stats: {e}")
    print("[memory] after dataset_train" + (", dataset_valid" if use_val else ""))
    _log_gpu_memory("after datasets")

    dataloader_train = DataLoader(
        dataset_train,
        shuffle=True,
        **cfg.dataloader,
        persistent_workers=True if cfg.dataloader.num_workers > 0 else False,
    )
    dataloader_valid = (
        DataLoader(
            dataset_valid,
            shuffle=False,
            **cfg.dataloader,
            persistent_workers=True if cfg.dataloader.num_workers > 0 else False,
        )
        if use_val
        else None
    )
    print("[memory] after dataloader_train" + (", dataloader_valid" if use_val else ""))
    _log_gpu_memory("after dataloaders")

    composer_model: ComposerModel = hydra.utils.instantiate(
        cfg.model,
        phase_conditioning=getattr(cfg, "phase_conditioning", None),
        phase_prediction=getattr(cfg, "phase_prediction", None),
        phase_rollout=getattr(cfg, "phase_rollout", None),
    )
    print("[memory] after composer_model = instantiate(cfg.model)")
    _log_gpu_memory("after model create")

    # Optional warm-start from an existing policy checkpoint (without Composer autoresume).
    resume_name = getattr(cfg, "resume_from_ckpt_name", None)
    if resume_name is not None:
        resume_episode = getattr(cfg, "resume_from_ckpt_episode", "latest")
        ckpt_dir = REPO_DIRS.CKPT / resume_name
        ckpt_path_list = list(ckpt_dir.glob(f"{resume_episode}*"))
        assert len(ckpt_path_list) > 0, f"No checkpoint found in {ckpt_dir} with {resume_episode}"
        assert len(ckpt_path_list) < 2, f"Multiple ckpts found in {ckpt_dir} with {resume_episode}"
        ckpt_fpath = ckpt_path_list[0]
        print(f"[resume] Loading model weights from {ckpt_fpath}")
        state_dict = torch.load(ckpt_fpath, map_location=DEVICE, weights_only=False)
        # Same layout as in FMPolicy.load_from_checkpoint: state['state']['model']
        _pcfg = getattr(cfg, "phase_conditioning", None)
        _ppred = getattr(cfg, "phase_prediction", None)
        _pe = bool(getattr(_pcfg, "enabled", False)) if _pcfg is not None else False
        _ppe = bool(getattr(_ppred, "enabled", False)) if _ppred is not None else False
        log_state_dict_load(
            composer_model,
            state_dict["state"]["model"],
            strict=False,
            tag="resume",
            phase_enabled=_pe,
            phase_prediction_enabled=_ppe,
        )
        print("[resume] Weights loaded into composer_model")

    optimizer = hydra.utils.instantiate(cfg.optimizer, composer_model.parameters())
    print("[memory] after optimizer")
    _log_gpu_memory("after optimizer")

    lr_scheduler = get_scheduler(
        cfg.lr_scheduler.name,
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_scheduler.num_warmup_steps,
        num_training_steps=(len(dataloader_train) * cfg.epochs),
        # pytorch assumes stepping LRScheduler every epoch
        # however huggingface diffusers steps it every batch
    )
    print("[memory] after lr_scheduler")
    _log_gpu_memory("after lr_scheduler")

    _log_memory_usage(cfg, composer_model, optimizer, dataset_train)

    # При log_wandb=False нельзя передавать WandBLogger(..., mode=disabled): Composer всё равно
    # дергает artifact download при autoresume → RuntimeError до init wandb.
    loggers = []
    if cfg.log_wandb:
        loggers.append(
            WandBLogger(
                project="pfp-train-fixed",
                entity="rl-lab-chisari",
                init_kwargs={
                    "config": OmegaConf.to_container(cfg),
                    "mode": "online",
                },
            )
        )
    print("[memory] after loggers setup")
    _log_gpu_memory("after loggers setup")

    print("[memory] >>> about to call Trainer(...)")
    _log_gpu_memory(">>> right before Trainer()")

    train_callbacks = [
        LRMonitor(),
        MemoryProfileCallback(),
        ResourceMonitorCallback(path_to_check=str(REPO_DIRS.ROOT)),
    ]
    _ms = getattr(cfg, "checkpoint_milestones", None)
    if _ms:
        train_callbacks.append(MilestoneCheckpointCopyCallback(list(OmegaConf.to_container(_ms, resolve=True))))

    save_every = int(cfg.save_each_n_epochs)
    save_keep = int(getattr(cfg, "save_num_checkpoints_to_keep", 5))
    max_ep = int(cfg.epochs)
    expected_ckpt_epochs = list(range(save_every, max_ep + 1, save_every))
    print(
        f"[checkpoint] save_interval={save_every}ep  save_num_checkpoints_to_keep={save_keep}  "
        f"expected files at epochs {expected_ckpt_epochs}"
    )
    if save_keep < len(expected_ckpt_epochs):
        print(
            f"[checkpoint] WARNING: save_num_checkpoints_to_keep={save_keep} < {len(expected_ckpt_epochs)} "
            f"milestone epochs — older checkpoints may be deleted during training.",
            file=sys.stderr,
        )

    trainer = Trainer(
        model=composer_model,
        train_dataloader=dataloader_train,
        eval_dataloader=dataloader_valid,
        max_duration=cfg.epochs,
        optimizers=optimizer,
        schedulers=lr_scheduler,
        step_schedulers_every_batch=True,
        device="gpu" if DEVICE.type == "cuda" else "cpu",
        loggers=loggers,
        callbacks=train_callbacks,
        save_folder="ckpt/{run_name}",
        save_interval=f"{save_every}ep",
        save_num_checkpoints_to_keep=save_keep,
        algorithms=[EMA()] if cfg.use_ema else None,
        run_name=cfg.run_name,
        # Full Composer autoresume (LR, optimizer, epoch) только при run_name и без resume_from_ckpt_name.
        # Работает с PyTorch 2.5; в 2.6+ torch.load(weights_only=True) ломает загрузку.
        autoresume=bool(cfg.run_name and not getattr(cfg, "resume_from_ckpt_name", None)),
        spin_dataloaders=False
    )
    print("[memory] <<< Trainer() returned")
    _log_gpu_memory("<<< after Trainer()")

    if cfg.log_wandb:
        wandb.watch(composer_model)
    # Save the used cfg for inference (same absolute path as checkpoints)
    ckpt_run_dir = REPO_DIRS.CKPT / trainer.state.run_name
    ckpt_run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, str(ckpt_run_dir / "config.yaml"))

    print("[memory] >>> about to call trainer.fit()")
    _log_gpu_memory(">>> before fit()")
    trainer.fit()
    print("[memory] <<< trainer.fit() returned")
    _log_gpu_memory("<<< after fit()")
    run_name = trainer.state.run_name
    if cfg.log_wandb:
        wandb.finish()
    trainer.close()

    if getattr(cfg, "launch_eval_after_train", True) and "CUDA_VISIBLE_DEVICES" in os.environ:
        _ = subprocess.Popen(
            [
                "bash",
                "bash/start_eval.sh",
                f"{os.environ['CUDA_VISIBLE_DEVICES']}",
                f"{run_name}",
            ],
            start_new_session=True,
        )
    return


if __name__ == "__main__":
    main()
