import os
import hydra
import wandb
import subprocess
import torch
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
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from pfp import DEVICE, DATA_DIRS, set_seeds
from pfp.data.dataset_pcd import RobotDatasetPcd
from pfp.data.dataset_images import RobotDatasetImages


ts.add_safe_globals([ListConfig, ContainerMetadata])

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

    data_path_train = DATA_DIRS.PFP / cfg.task_name / "train"
    data_path_valid = DATA_DIRS.PFP / cfg.task_name / "valid"
    if cfg.obs_mode == "pcd":
        dataset_train = RobotDatasetPcd(data_path_train, **cfg.dataset)
        dataset_valid = RobotDatasetPcd(data_path_valid, **cfg.dataset)
    elif cfg.obs_mode == "rgb":
        dataset_train = RobotDatasetImages(data_path_train, **cfg.dataset)
        dataset_valid = RobotDatasetImages(data_path_valid, **cfg.dataset)
    else:
        raise ValueError(f"Unknown observation mode: {cfg.obs_mode}")
    print("[memory] after dataset_train, dataset_valid")
    _log_gpu_memory("after datasets")

    dataloader_train = DataLoader(
        dataset_train,
        shuffle=True,
        **cfg.dataloader,
        persistent_workers=True if cfg.dataloader.num_workers > 0 else False,
    )
    dataloader_valid = DataLoader(
        dataset_valid,
        shuffle=False,
        **cfg.dataloader,
        persistent_workers=True if cfg.dataloader.num_workers > 0 else False,
    )
    print("[memory] after dataloader_train, dataloader_valid")
    _log_gpu_memory("after dataloaders")

    composer_model: ComposerModel = hydra.utils.instantiate(cfg.model)
    print("[memory] after composer_model = instantiate(cfg.model)")
    _log_gpu_memory("after model create")

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

    wandb_logger = WandBLogger(
        project="pfp-train-fixed",
        entity="rl-lab-chisari",
        init_kwargs={
            "config": OmegaConf.to_container(cfg),
            "mode": "online" if cfg.log_wandb else "disabled",
        },
    )
    print("[memory] after wandb_logger")
    _log_gpu_memory("after wandb_logger")

    print("[memory] >>> about to call Trainer(...)")
    _log_gpu_memory(">>> right before Trainer()")

    # Патч: логируем первый и следующие вызовы tensor_to_device внутри Composer (там падает OOM).
    try:
        import composer.devices.device_gpu as _dg
        _orig = _dg.DeviceGPU.tensor_to_device
        _call_count = [0]

        def _logged_tensor_to_device(self, tensor):
            _call_count[0] += 1
            print(f"[memory] DeviceGPU.tensor_to_device call #{_call_count[0]} tensor.shape={tuple(tensor.shape)} dtype={tensor.dtype}")
            _log_gpu_memory("  before tensor_to_device")
            out = _orig(self, tensor)
            _log_gpu_memory("  after tensor_to_device")
            return out

        _dg.DeviceGPU.tensor_to_device = _logged_tensor_to_device
        print("[memory] Patched DeviceGPU.tensor_to_device for debug")
    except Exception as e:
        print(f"[memory] Could not patch DeviceGPU: {e}")

    trainer = Trainer(
        model=composer_model,
        train_dataloader=dataloader_train,
        eval_dataloader=dataloader_valid,
        max_duration=cfg.epochs,
        optimizers=optimizer,
        schedulers=lr_scheduler,
        step_schedulers_every_batch=True,
        device="gpu" if DEVICE.type == "cuda" else "cpu",
        loggers=[wandb_logger],
        callbacks=[LRMonitor(), MemoryProfileCallback()],
        save_folder="ckpt/{run_name}",
        save_interval=f"{cfg.save_each_n_epochs}ep",
        save_num_checkpoints_to_keep=3,
        algorithms=[EMA()] if cfg.use_ema else None,
        run_name=cfg.run_name,  # set this to continue training from previous ckpt
        autoresume=True if cfg.run_name is not None else False,
        spin_dataloaders=False
    )
    print("[memory] <<< Trainer() returned")
    _log_gpu_memory("<<< after Trainer()")

    wandb.watch(composer_model)
    # Save the used cfg for inference
    OmegaConf.save(cfg, "ckpt/" + trainer.state.run_name + "/config.yaml")

    print("[memory] >>> about to call trainer.fit()")
    _log_gpu_memory(">>> before fit()")
    trainer.fit()
    print("[memory] <<< trainer.fit() returned")
    _log_gpu_memory("<<< after fit()")
    run_name = trainer.state.run_name
    wandb.finish()
    trainer.close()

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
