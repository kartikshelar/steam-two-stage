"""Minimum detectable effect / sample-size calculator.

Traffic numbers below are *stated assumptions for a hypothetical storefront*.
They are not this dump's users and not production traffic.
"""

from __future__ import annotations

from math import ceil

from scipy.stats import norm


def n_per_arm_two_proportion(
    p: float,
    delta: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """Users per arm for a two-sided two-proportion z-test, equal allocation.

    `p` is the control conversion rate. `delta` is the absolute lift to detect
    (treatment = p + delta).
    """
    if delta <= 0:
        raise ValueError("delta must be positive")
    if not (0 < p < 1):
        raise ValueError("p must be in (0, 1)")
    p2 = p + delta
    if p2 >= 1:
        raise ValueError("p + delta must be < 1")
    z_a = float(norm.ppf(1.0 - alpha / 2.0))
    z_b = float(norm.ppf(power))
    n = (z_a + z_b) ** 2 * (p * (1.0 - p) + p2 * (1.0 - p2)) / (delta ** 2)
    return int(ceil(n))


def days_for_test(
    n_per_arm: int,
    daily_active_users: int,
) -> int:
    """50/50 split, one observation per user per day. Both arms need n_per_arm."""
    if daily_active_users <= 0:
        raise ValueError("daily_active_users must be positive")
    per_arm_per_day = daily_active_users / 2.0
    return int(ceil(n_per_arm / per_arm_per_day))


def mde_table(
    p: float,
    relative_lifts: list[float],
    daily_active_users: list[int],
    alpha: float = 0.05,
    power: float = 0.8,
) -> list[dict]:
    rows = []
    for rel in relative_lifts:
        delta = p * rel
        n = n_per_arm_two_proportion(p, delta, alpha=alpha, power=power)
        for dau in daily_active_users:
            rows.append(
                {
                    "baseline_p": p,
                    "relative_lift": rel,
                    "absolute_lift": delta,
                    "alpha": alpha,
                    "power": power,
                    "n_per_arm": n,
                    "n_total": 2 * n,
                    "daily_active_users": dau,
                    "days": days_for_test(n, dau),
                }
            )
    return rows
