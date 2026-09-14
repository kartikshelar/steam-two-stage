from __future__ import annotations

import pandas as pd

from eval.coldstart import cold_slices


def test_cold_slices_partition():
    test = pd.DataFrame(
        {
            "user_id": ["warm_u", "cold_u", "warm_u", "cold_u"],
            "item_id": ["warm_i", "warm_i", "cold_i", "cold_i"],
        }
    )
    slices = cold_slices(test, keep_users={"warm_u"}, keep_items={"warm_i"})
    assert list(slices["warm"]["user_id"]) == ["warm_u"]
    assert set(slices["cold_user"]["user_id"]) == {"cold_u"}
    assert set(slices["cold_item"]["item_id"]) == {"cold_i"}
    assert len(slices["cold_user_or_item"]) == 3
