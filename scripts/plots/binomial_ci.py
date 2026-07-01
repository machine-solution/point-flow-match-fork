"""Binomial confidence intervals for success-rate plots."""
from __future__ import annotations

from scipy.stats import beta


def clopper_pearson(k: int, n: int, alpha: float) -> tuple[float, float]:
    """Exact central binomial CI for proportion k/n (Clopper–Pearson)."""
    if n <= 0:
        return 0.0, 0.0
    k = int(k)
    n = int(n)
    if k == 0:
        lo = 0.0
    else:
        lo = float(beta.ppf(alpha / 2, k, n - k + 1))
    if k == n:
        hi = 1.0
    else:
        hi = float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi
