#!/usr/bin/env python3
"""Benchmark one training step (forward+backward+optimizer)."""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import hydra
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

_diffusion_policy_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "diffusion_policy")
if os.path.exists(_diffusion_policy_path) and _diffusion_policy_path not in sys.path:
    sys.path.insert(0, _diffusion_policy_path)

from pfp import DATA_DIRS, DEVICE, REPO_DIRS, set_seeds
from pfp.data.dataset_images import RobotDatasetImages
from pfp.data.dataset_pcd import RobotDatasetPcd


class _NoopLogger:
    def log_metrics(self, *_args, **_kwargs):
        return


def _load_train_cfg(path: Path) -> OmegaConf:
    path = path.expanduser().resolve()
    conf_dir = (Path(__file__).resolve().parents[1] / "conf").resolve()
    if path == (conf_dir / "train.yaml"):
        with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
            cfg = compose(config_name="train")
        return cfg
    return OmegaConf.load(path)


def _resolve_dataset_path(cfg: OmegaConf, *, train: bool) -> Path:
    key = "dataset_path_train" if train else "dataset_path_valid"
    override = getattr(cfg, key, None)
    if override is not None:
        s = str(override).strip()
        if s and s.lower() not in ("null", "none", "~"):
            return Path(s).expanduser().resolve()
    sub = "train" if train else "valid"
    return (DATA_DIRS.PFP / cfg.task_name / sub).resolve()


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _to_device(batch):
    out = []
    for x in batch:
        if isinstance(x, torch.Tensor):
            out.append(x.to(DEVICE))
        else:
            out.append(x)
    return tuple(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-config", type=Path, default=Path("conf/train.yaml"))
    ap.add_argument("--batch-size", type=int, default=None, help="Override dataloader.batch_size")
    ap.add_argument("--seed", type=int, default=5678)
    ap.add_argument("--output-csv", type=Path, default=Path("results/efficiency/train_step.csv"))
    args = ap.parse_args()

    set_seeds(args.seed)
    cfg = _load_train_cfg(args.train_config)
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    if args.batch_size is not None:
        cfg.dataloader.batch_size = int(args.batch_size)

    data_path = _resolve_dataset_path(cfg, train=True)
    if not data_path.exists():
        fallback_task = "open_fridge"
        fallback_path = (DATA_DIRS.PFP / fallback_task / "train").resolve()
        if str(cfg.task_name) != fallback_task and fallback_path.exists():
            print(
                f"[benchmark_train_step] dataset for task={cfg.task_name} not found at {data_path}; "
                f"falling back to task={fallback_task}"
            )
            cfg.task_name = fallback_task
            data_path = fallback_path
        else:
            raise FileNotFoundError(
                f"Dataset path not found: {data_path}. Set dataset_path_train or ensure data is present."
            )
    if cfg.obs_mode == "pcd":
        ds = RobotDatasetPcd(data_path, phase_conditioning=getattr(cfg, "phase_conditioning", None), **cfg.dataset)
    elif cfg.obs_mode == "rgb":
        ds = RobotDatasetImages(data_path, phase_conditioning=getattr(cfg, "phase_conditioning", None), **cfg.dataset)
    else:
        raise ValueError(f"Unsupported obs_mode: {cfg.obs_mode}")

    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=int(cfg.dataloader.batch_size),
        shuffle=True,
        num_workers=0,
    )
    batch = _to_device(next(iter(dl)))

    model = hydra.utils.instantiate(
        cfg.model,
        phase_conditioning=getattr(cfg, "phase_conditioning", None),
        phase_prediction=getattr(cfg, "phase_prediction", None),
        phase_rollout=getattr(cfg, "phase_rollout", None),
    )
    model.logger = _NoopLogger()
    print(f"[model] class={model.__class__.__name__} target={cfg.model._target_}")
    print(f"[model] num_k_infer={getattr(model, 'num_k_infer', None)}")
    if hasattr(model, "meanflow_enabled"):
        print(
            f"[model] meanflow.enabled={getattr(model, 'meanflow_enabled', None)} "
            f"one_step={getattr(model, 'meanflow_one_step', None)} "
            f"interval_embed_dim={getattr(model, 'interval_embed_dim', None)}"
        )
    model.to(DEVICE)
    model.train()
    optimizer = hydra.utils.instantiate(cfg.optimizer, model.parameters())

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    _sync()
    t0 = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    loss = model.loss(model.forward(batch), batch)
    loss.backward()
    optimizer.step()
    _sync()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    max_gpu_memory_mb = (
        float(torch.cuda.max_memory_allocated() / (1024 ** 2)) if torch.cuda.is_available() else None
    )

    row = {
        "train_config": str(args.train_config),
        "task_name": str(cfg.task_name),
        "obs_mode": str(cfg.obs_mode),
        "batch_size": int(cfg.dataloader.batch_size),
        "step_time_ms": float(elapsed_ms),
        "loss": float(loss.item()),
        "max_gpu_memory_mb": max_gpu_memory_mb,
    }
    out_csv = args.output_csv.expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_csv.exists()
    with out_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(f"Wrote: {out_csv}")
    print(
        f"Train step: {row['step_time_ms']:.2f} ms | loss={row['loss']:.4f} | "
        f"max_gpu_memory_mb={row['max_gpu_memory_mb']}"
    )


if __name__ == "__main__":
    main()
