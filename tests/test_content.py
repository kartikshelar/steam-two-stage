from __future__ import annotations

import pandas as pd

from retrieval.content import _parse_list, fit_content_vocab, build_content_tables


def test_parse_list_unescapes_amp():
    assert "Design & Illustration" in _parse_list('["Design &amp; Illustration", "Indie"]')


def test_content_tables_align_to_item_idx():
    games = pd.DataFrame(
        {
            "item_id": ["a", "b"],
            "genres": ['["Action"]', '["Indie"]'],
            "tags": ['["Action", "Multiplayer"]', '["Indie"]'],
            "price": [9.99, 0.0],
            "release_date": pd.to_datetime(["2011-01-01", "2016-06-01"], utc=True),
        }
    )
    vocab = fit_content_vocab(games, {"a", "b"})
    item_to_idx = {"a": 1, "b": 2}
    tables = build_content_tables(item_to_idx, games, vocab)
    assert tables["genre_ids"].shape[0] == 3
    assert tables["genre_ids"][0].sum() == 0
    assert tables["genre_ids"][1].sum() > 0
    assert tables["numeric"][1, 1] == 0.0  # price present
