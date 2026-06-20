"""Helpers for rollout inference timing breakdown."""
from __future__ import annotations

import time
from typing import Callable, TypeVar

import torch

T = TypeVar("T")

ACTION_PROFILE_KEYS = (
    "update_obs_lists_ms",
    "sample_stacked_obs_ms",
    "numpy_to_torch_ms",
    "normalization_ms",
    "infer_y_ms",
    "denormalization_ms",
    "total_predict_action_ms",
)

INFER_Y_PROFILE_KEYS = (
    "encode_obs_ms",
    "scheduler_ms",
    "loop_total_ms",
    "unet_total_ms",
    "extra_mlp_ms",
    "clone_detach_ms",
    "other_loop_ms",
    "nfe",
)


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def blank_action_profile() -> dict[str, float]:
    return {k: 0.0 for k in ACTION_PROFILE_KEYS}


def blank_infer_y_profile() -> dict[str, float]:
    out = {k: 0.0 for k in INFER_Y_PROFILE_KEYS}
    out["nfe"] = 0.0
    return out


def merge_action_profile(action: dict[str, float], infer_y: dict[str, float]) -> dict[str, float]:
    out = dict(action)
    out["encode_obs_ms"] = float(infer_y.get("encode_obs_ms", 0.0))
    out["infer_y_ms"] = float(action.get("infer_y_ms", 0.0))
    out["nfe"] = float(infer_y.get("nfe", 0.0))
    out["unet_total_ms"] = float(infer_y.get("unet_total_ms", 0.0))
    unet_total = out["unet_total_ms"]
    nfe = out["nfe"]
    out["unet_ms_per_nfe"] = float(unet_total / nfe) if nfe > 0 else 0.0
    out["scheduler_ms"] = float(infer_y.get("scheduler_ms", 0.0))
    out["extra_mlp_ms"] = float(infer_y.get("extra_mlp_ms", 0.0))
    out["clone_detach_ms"] = float(infer_y.get("clone_detach_ms", 0.0))
    out["other_loop_ms"] = float(infer_y.get("other_loop_ms", 0.0))
    out["loop_total_ms"] = float(infer_y.get("loop_total_ms", 0.0))
    return out


def timed_call(profile: dict[str, float], key: str, fn: Callable[[], T]) -> T:
    cuda_sync()
    t0 = time.perf_counter()
    out = fn()
    cuda_sync()
    profile[key] = profile.get(key, 0.0) + (time.perf_counter() - t0) * 1000.0
    return out


def add_elapsed_ms(profile: dict[str, float], key: str, t0: float) -> None:
    cuda_sync()
    profile[key] = profile.get(key, 0.0) + (time.perf_counter() - t0) * 1000.0
