"""Inference time grids for multi-step MeanFlow."""
from __future__ import annotations

import math

import torch

from pfp.common.fm_utils import get_timesteps

MEANFLOW_INFER_SCHEDULES = (
    "uniform",
    "fm_exp",
    "reverse_exp",
    "cosine",
    "beta_2_2",
    "beta_3_3",
)


def _finalize_grid(dt: torch.Tensor, *, device, dtype) -> torch.Tensor:
    dt = dt.to(device=device, dtype=dtype)
    if torch.any(dt <= 0):
        raise ValueError(f"Non-positive dt in meanflow grid: min={float(dt.min())}")
    dt = dt / dt.sum()
    grid = torch.cat([torch.zeros(1, device=device, dtype=dtype), torch.cumsum(dt, dim=0)])
    _assert_valid_meanflow_grid(grid)
    return grid


def _assert_valid_meanflow_grid(grid: torch.Tensor) -> None:
    if grid.ndim != 1:
        raise ValueError(f"meanflow grid must be 1D, got shape={tuple(grid.shape)}")
    if grid.numel() < 2:
        raise ValueError("meanflow grid must have at least 2 points")
    g0 = float(grid[0].item())
    g1 = float(grid[-1].item())
    if abs(g0) > 1e-6:
        raise ValueError(f"meanflow grid must start at 0, got {g0}")
    if abs(g1 - 1.0) > 1e-5:
        raise ValueError(f"meanflow grid must end at 1, got {g1}")
    if not torch.all(grid[1:] > grid[:-1]):
        raise ValueError("meanflow grid must be strictly increasing")


def build_meanflow_time_grid(
    k: int,
    schedule: str,
    *,
    exp_scale: float = 4.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return monotonic grid [t_0..t_K] with t_0=0 and t_K=1."""
    k = int(k)
    if k <= 0:
        raise ValueError("k must be >= 1")
    schedule = str(schedule)
    device = device if device is not None else torch.device("cpu")

    if schedule == "uniform":
        grid = torch.linspace(0.0, 1.0, k + 1, device=device, dtype=dtype)
        _assert_valid_meanflow_grid(grid)
        return grid

    if schedule == "fm_exp":
        _, dt = get_timesteps("exp", k, exp_scale=float(exp_scale))
        return _finalize_grid(dt, device=device, dtype=dtype)

    if schedule == "reverse_exp":
        grid_exp = build_meanflow_time_grid(
            k, "fm_exp", exp_scale=exp_scale, device=device, dtype=dtype
        )
        grid = 1.0 - torch.flip(grid_exp, dims=[0])
        _assert_valid_meanflow_grid(grid)
        return grid

    if schedule == "cosine":
        s = torch.linspace(0.0, 1.0, k + 1, device=device, dtype=dtype)
        grid = 0.5 - 0.5 * torch.cos(math.pi * s)
        _assert_valid_meanflow_grid(grid)
        return grid

    if schedule in ("beta_2_2", "beta_3_3"):
        m = (torch.arange(k, device=device, dtype=dtype) + 0.5) / float(k)
        if schedule == "beta_2_2":
            w = m * (1.0 - m)
        else:
            w = (m ** 2) * ((1.0 - m) ** 2)
        return _finalize_grid(w, device=device, dtype=dtype)

    raise ValueError(
        f"Unknown meanflow schedule: {schedule}. Expected one of {MEANFLOW_INFER_SCHEDULES}"
    )


def meanflow_grid_deltas(grid: torch.Tensor) -> list[float]:
    dt = grid[1:] - grid[:-1]
    return [float(x.item()) for x in dt]


def meanflow_grid_payload(grid: torch.Tensor) -> dict:
    grid_list = [float(x.item()) for x in grid]
    dt_list = meanflow_grid_deltas(grid)
    return {
        "time_grid": grid_list,
        "dt_grid": dt_list,
        "dt_max_idx": int(torch.argmax(grid[1:] - grid[:-1]).item()) if len(dt_list) else 0,
    }
