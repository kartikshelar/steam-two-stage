from __future__ import annotations

from data.prepare import parse_mcauley_line, parse_price


def test_parse_python_literal_review():
    line = "{'username': 'Chaos Syren', 'hours': 0.1, 'product_id': '725280', 'date': '2017-12-17'}"
    obj = parse_mcauley_line(line)
    assert obj["username"] == "Chaos Syren"
    assert obj["hours"] == 0.1
    assert obj["product_id"] == "725280"


def test_parse_json_review():
    line = '{"username": "x", "hours": 2.5, "product_id": "1", "date": "2017-01-01"}'
    obj = parse_mcauley_line(line)
    assert obj["hours"] == 2.5


def test_parse_price_free_and_numeric():
    assert parse_price("Free to Play") == 0.0
    assert parse_price(19.99) == 19.99
    assert parse_price("$4.99") == 4.99
