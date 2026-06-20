"""Time grids and sampling helpers for Momentum MeanFlow."""
from __future__ import annotations

import torch

from pfp.common.fm_utils import get_timesteps

MOMENTUM_INFER_SCHEDULES = ("uniform", "fm_exp", "cosine", "beta_2_2", "small_end")


def build_inference_time_grid(
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
    if schedule == "uniform":
        return torch.linspace(0.0, 1.0, k + 1, device=device, dtype=dtype)

    if schedule in ("fm_exp", "linear", "cosine", "exp"):
        sched = "exp" if schedule == "fm_exp" else schedule
        _, dt = get_timesteps(sched, k, exp_scale=float(exp_scale))
        dt = dt.to(device=device, dtype=dtype)
        grid = [torch.zeros((), device=device, dtype=dtype)]
        acc = torch.zeros((), device=device, dtype=dtype)
        for d in dt:
            acc = acc + d
            grid.append(acc)
        return torch.stack(grid)

    if schedule == "beta_2_2":
        t = torch.linspace(0.0, 1.0, k + 1, device=device, dtype=dtype)[:-1]
        w = (t + 1e-8) ** 2
        dt = w / w.sum()
        grid = [torch.zeros((), device=device, dtype=dtype)]
        acc = torch.zeros((), device=device, dtype=dtype)
        for d in dt:
            acc = acc + d
            grid.append(acc)
        return torch.stack(grid)

    if schedule == "small_end":
        idx = torch.arange(k, device=device, dtype=dtype) + 1.0
        w = 1.0 / idx
        dt = w / w.sum()
        grid = [torch.zeros((), device=device, dtype=dtype)]
        acc = torch.zeros((), device=device, dtype=dtype)
        for d in dt:
            acc = acc + d
            grid.append(acc)
        return torch.stack(grid)

    raise ValueError(f"Unknown momentum_meanflow schedule: {schedule}")


def _resample_sorted_times(num: int, batch_size: int, dt_min: float, device: torch.device) -> torch.Tensor:
    dt_min = float(dt_min)
    out = torch.empty((batch_size, num), device=device)
    for b in range(batch_size):
        ok = False
        for _ in range(64):
            times = torch.rand(num, device=device).sort().values
            if torch.all(times[1:] - times[:-1] >= dt_min):
                out[b] = times
                ok = True
                break
        if not ok:
            # Fallback: uniform grid with jitter.
            base = torch.linspace(0.0, 1.0, num + 2, device=device)[1:-1]
            out[b] = base
    return out


def sample_two_times(batch_size: int, *, dt_min: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    times = _resample_sorted_times(2, batch_size, dt_min, device)
    t0 = times[:, 0:1].unsqueeze(-1)
    t1 = times[:, 1:2].unsqueeze(-1)
    return t0, t1


def sample_three_times(
    batch_size: int, *, dt_min: float, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    times = _resample_sorted_times(3, batch_size, dt_min, device)
    t0 = times[:, 0:1].unsqueeze(-1)
    t1 = times[:, 1:2].unsqueeze(-1)
    t2 = times[:, 2:3].unsqueeze(-1)
    return t0, t1, t2
