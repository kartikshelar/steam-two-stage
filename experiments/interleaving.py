"""Team-draft interleaving. Offline clicks = ground-truth items in the slate.

This is not a live interleaving experiment. It asks: on the same users, if we
merged two rankings and treated held-out purchases as clicks, which team
collects more credit?
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def team_draft(
    list_a: list[str],
    list_b: list[str],
    k: int,
    rng: np.random.RandomState,
) -> tuple[list[str], list[str]]:
    """Interleave A and B to length k without duplicates.

    When both teams still have unused items, a fair coin picks the next team.
    """
    ia = ib = 0
    out: list[str] = []
    teams: list[str] = []
    seen: set[str] = set()
    while len(out) < k:
        while ia < len(list_a) and list_a[ia] in seen:
            ia += 1
        while ib < len(list_b) and list_b[ib] in seen:
            ib += 1
        a_has = ia < len(list_a)
        b_has = ib < len(list_b)
        if not a_has and not b_has:
            break
        if a_has and b_has:
            pick_a = bool(rng.rand() < 0.5)
        else:
            pick_a = a_has
        if pick_a:
            item = list_a[ia]
            ia += 1
            team = "A"
        else:
            item = list_b[ib]
            ib += 1
            team = "B"
        seen.add(item)
        out.append(item)
        teams.append(team)
    return out, teams


def credit_clicks(items: list[str], teams: list[str], clicks: Iterable[str]) -> tuple[int, int]:
    click_set = set(clicks)
    a = b = 0
    for item, team in zip(items, teams):
        if item not in click_set:
            continue
        if team == "A":
            a += 1
        elif team == "B":
            b += 1
    return a, b


def compare_rankings(
    ranked_a: dict[str, list[str]],
    ranked_b: dict[str, list[str]],
    truth: dict[str, set],
    users: list[str],
    k: int,
    rng: np.random.RandomState,
) -> dict[str, float]:
    n_a = n_b = n_tie = 0
    n_users = 0
    credit_a = credit_b = 0
    for user in users:
        clicks = truth.get(user) or set()
        if not clicks:
            continue
        items, teams = team_draft(ranked_a[user][:k], ranked_b[user][:k], k, rng)
        ca, cb = credit_clicks(items, teams, clicks)
        credit_a += ca
        credit_b += cb
        n_users += 1
        if ca > cb:
            n_a += 1
        elif cb > ca:
            n_b += 1
        else:
            n_tie += 1
    decisive = n_a + n_b
    p_a = n_a / decisive if decisive else float("nan")
    # Two-sided exact binomial under P(A wins decisive)=0.5, using normal approx with continuity.
    if decisive > 0:
        se = np.sqrt(0.25 / decisive)
        z = (p_a - 0.5) / se
        # two-sided p from |z|
        from math import erfc

        p_value = float(erfc(abs(z) / np.sqrt(2.0)))
    else:
        z = float("nan")
        p_value = float("nan")
    return {
        "n_users": float(n_users),
        "n_a_wins": float(n_a),
        "n_b_wins": float(n_b),
        "n_ties": float(n_tie),
        "p_a_win_given_decisive": float(p_a),
        "credit_a": float(credit_a),
        "credit_b": float(credit_b),
        "z_a_vs_b": float(z),
        "p_value_two_sided": float(p_value),
        "k": float(k),
    }
