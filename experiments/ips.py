"""Inverse-propensity and self-normalized IPS. Offline only — not an A/B test.

Logging and target policies share a candidate set. Propensity is a rank-softmax:
π(item) ∝ exp(-rank / temperature), rank 0 = best. That puts two-tower inner
products and GBDT scores on the same scale.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def rank_softmax(n: int, temperature: float) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    ranks = np.arange(n, dtype=np.float64)
    z = -ranks / max(float(temperature), 1e-8)
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


def sample_from_ranking(
    ranked: list[str],
    temperature: float,
    rng: np.random.RandomState,
) -> tuple[str, np.ndarray]:
    p = rank_softmax(len(ranked), temperature)
    idx = int(rng.choice(len(ranked), p=p))
    return ranked[idx], p


def _weights(p_target: np.ndarray, p_log: np.ndarray, clip: float | None) -> tuple[np.ndarray, np.ndarray]:
    p_log_safe = np.clip(p_log, 1e-12, None)
    raw = p_target / p_log_safe
    if clip is None:
        return raw, raw
    return np.minimum(raw, float(clip)), raw


def ips_snips(
    rewards: np.ndarray,
    p_target: np.ndarray,
    p_log: np.ndarray,
    clip: float | None = 20.0,
) -> dict[str, float]:
    rewards = np.asarray(rewards, dtype=np.float64)
    p_target = np.asarray(p_target, dtype=np.float64)
    p_log = np.asarray(p_log, dtype=np.float64)
    w, raw = _weights(p_target, p_log, clip)
    n = int(len(rewards))
    if n == 0:
        raise ValueError("empty IPS sample")
    ips_values = rewards * w
    ips = float(ips_values.mean())
    ips_var = float(ips_values.var(ddof=1)) if n > 1 else 0.0
    ips_se = float(np.sqrt(ips_var / n)) if n > 0 else float("nan")
    w_sum = float(w.sum())
    snips = float((rewards * w).sum() / w_sum) if w_sum > 0 else float("nan")
    ess = float((w_sum ** 2) / float((w ** 2).sum())) if w_sum > 0 else 0.0
    return {
        "n": float(n),
        "ips": ips,
        "ips_var": ips_var,
        "ips_se": ips_se,
        "ips_ci95_lo": ips - 1.96 * ips_se,
        "ips_ci95_hi": ips + 1.96 * ips_se,
        "snips": snips,
        "ess": ess,
        "mean_weight": float(w.mean()),
        "max_weight": float(w.max()) if n else 0.0,
        "max_weight_unclipped": float(raw.max()) if n else 0.0,
        "clip": float("nan") if clip is None else float(clip),
        "clip_frac": float((raw > float(clip)).mean()) if clip is not None else 0.0,
        "mean_reward": float(rewards.mean()),
    }


def bootstrap_mean_ci(
    values: np.ndarray,
    rng: np.random.RandomState,
    n_boot: int = 1000,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.randint(0, n, n)
        means[i] = values[idx].mean()
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi), float(means.std(ddof=1))


def bootstrap_snips_ci(
    rewards: np.ndarray,
    weights: np.ndarray,
    rng: np.random.RandomState,
    n_boot: int = 1000,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    rewards = np.asarray(rewards, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    n = len(rewards)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    stats = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.randint(0, n, n)
        denom = weights[idx].sum()
        stats[i] = (rewards[idx] * weights[idx]).sum() / denom if denom > 0 else np.nan
    finite = stats[np.isfinite(stats)]
    if len(finite) == 0:
        return float("nan"), float("nan"), float("nan")
    lo, hi = np.quantile(finite, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi), float(finite.std(ddof=1))


def simulate_bandit_logs(
    ranked_log: dict[str, list[str]],
    ranked_target: dict[str, list[str]],
    truth: dict[str, set],
    users: list[str],
    temperature: float,
    rng: np.random.RandomState,
) -> dict[str, np.ndarray]:
    """One sampled impression per user from the logging ranking."""
    actions: list[str] = []
    rewards: list[float] = []
    p_log: list[float] = []
    p_tgt: list[float] = []
    for user in users:
        log_list = ranked_log[user]
        tgt_list = ranked_target[user]
        if not log_list or not tgt_list:
            continue
        action, p0 = sample_from_ranking(log_list, temperature, rng)
        p1_map = {item: p for item, p in zip(tgt_list, rank_softmax(len(tgt_list), temperature))}
        actions.append(action)
        rewards.append(1.0 if action in truth.get(user, set()) else 0.0)
        p_log.append(float(p0[log_list.index(action)]))
        p_tgt.append(float(p1_map.get(action, 0.0)))
    return {
        "reward": np.asarray(rewards, dtype=np.float64),
        "p_log": np.asarray(p_log, dtype=np.float64),
        "p_target": np.asarray(p_tgt, dtype=np.float64),
    }


def estimate_target_from_logs(
    logs: dict[str, np.ndarray],
    clip: float | None,
    rng: np.random.RandomState,
    n_boot: int,
) -> dict[str, Any]:
    est = ips_snips(logs["reward"], logs["p_target"], logs["p_log"], clip=clip)
    w, _ = _weights(logs["p_target"], logs["p_log"], clip)
    ips_lo, ips_hi, ips_boot_se = bootstrap_mean_ci(logs["reward"] * w, rng, n_boot=n_boot)
    sn_lo, sn_hi, sn_boot_se = bootstrap_snips_ci(logs["reward"], w, rng, n_boot=n_boot)
    est["ips_boot_ci95_lo"] = ips_lo
    est["ips_boot_ci95_hi"] = ips_hi
    est["ips_boot_se"] = ips_boot_se
    est["snips_boot_ci95_lo"] = sn_lo
    est["snips_boot_ci95_hi"] = sn_hi
    est["snips_boot_se"] = sn_boot_se
    return est
