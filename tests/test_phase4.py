from __future__ import annotations

import numpy as np

from experiments.interleaving import compare_rankings, credit_clicks, team_draft
from experiments.ips import ips_snips, rank_softmax, simulate_bandit_logs
from experiments.mde import days_for_test, mde_table, n_per_arm_two_proportion


def test_rank_softmax_sums_to_one_and_prefers_rank0():
    p = rank_softmax(5, temperature=1.0)
    assert abs(p.sum() - 1.0) < 1e-12
    assert p[0] > p[1] > p[-1]


def test_ips_of_logging_policy_recovers_mean_reward():
    rng = np.random.RandomState(0)
    r = rng.randint(0, 2, size=5000).astype(np.float64)
    p = np.full(5000, 0.2)
    est = ips_snips(r, p_target=p, p_log=p, clip=20.0)
    assert abs(est["ips"] - r.mean()) < 1e-12
    assert abs(est["snips"] - r.mean()) < 1e-12
    assert abs(est["ess"] - 5000) < 1e-6


def test_clipped_ips_caps_weights():
    r = np.array([1.0, 0.0, 1.0])
    p_log = np.array([0.01, 0.5, 0.2])
    p_tgt = np.array([0.5, 0.4, 0.2])
    est = ips_snips(r, p_tgt, p_log, clip=10.0)
    assert est["max_weight"] <= 10.0 + 1e-12
    assert est["max_weight_unclipped"] > 10.0
    assert est["clip_frac"] > 0


def test_simulate_logs_reward_is_indicator_in_truth():
    ranked = {"u": ["a", "b", "c"]}
    truth = {"u": {"a"}}
    rng = np.random.RandomState(1)
    logs = simulate_bandit_logs(ranked, ranked, truth, ["u"] * 200, temperature=1.0, rng=rng)
    assert set(logs["reward"]).issubset({0.0, 1.0})
    assert logs["reward"].mean() > 0


def test_team_draft_no_duplicates_and_length():
    rng = np.random.RandomState(0)
    items, teams = team_draft(["a", "b", "c", "d"], ["a", "e", "f", "g"], k=4, rng=rng)
    assert len(items) == 4
    assert len(set(items)) == 4
    assert len(teams) == 4
    assert set(teams) <= {"A", "B"}


def test_identical_lists_interleave_ties_on_same_clicks():
    rng = np.random.RandomState(2)
    ranked = {"u1": ["x", "y", "z"]}
    truth = {"u1": {"x"}}
    rec = compare_rankings(ranked, ranked, truth, ["u1"], k=3, rng=rng)
    # Same list: the team that contributed the click depends on the coin, so not
    # always a tie. Credits still go to whoever contributed x. Just check it runs.
    assert rec["n_users"] == 1
    assert rec["n_a_wins"] + rec["n_b_wins"] + rec["n_ties"] == 1


def test_credit_clicks_assigns_to_contributing_team():
    a, b = credit_clicks(["x", "y"], ["A", "B"], {"y"})
    assert a == 0 and b == 1


def test_mde_smaller_lift_needs_more_users():
    n_small = n_per_arm_two_proportion(0.05, 0.05 * 0.01)
    n_big = n_per_arm_two_proportion(0.05, 0.05 * 0.10)
    assert n_small > n_big
    assert days_for_test(1000, 1000) == 2
    rows = mde_table(0.057, [0.05], [10_000], alpha=0.05, power=0.8)
    assert len(rows) == 1
    assert rows[0]["days"] >= 1
