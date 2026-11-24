"""Pure tests for reciprocal_rank_fusion."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.retrieval import reciprocal_rank_fusion  # noqa: E402


def test_ordering_and_scores() -> None:
    out = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]], k=60)
    ids = [c for c, _ in out]
    assert ids[:2] == ["a", "b"]  # equal scores; a wins on id/best-rank tie-break
    assert dict(out)["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert dict(out)["d"] == pytest.approx(1 / 63)
    assert dict(out)["c"] == pytest.approx(1 / 63)
    assert ids[-2:] == ["c", "d"]


def test_item_in_both_lists_beats_single_list_top() -> None:
    out = reciprocal_rank_fusion([["x", "shared"], ["y", "shared"]])
    assert out[0][0] == "shared"


def test_k_effect() -> None:
    small = dict(reciprocal_rank_fusion([["a", "b"]], k=1))
    large = dict(reciprocal_rank_fusion([["a", "b"]], k=1000))
    assert small["a"] / small["b"] > large["a"] / large["b"]
    assert small["a"] == pytest.approx(1 / 2)


def test_ties_are_deterministic() -> None:
    a = reciprocal_rank_fusion([["z", "m"], ["a", "q"]])
    b = reciprocal_rank_fusion([["a", "q"], ["z", "m"]])
    assert [c for c, _ in a] == [c for c, _ in b] == ["a", "z", "m", "q"]


def test_disjoint_lists() -> None:
    out = reciprocal_rank_fusion([["a", "b"], ["c", "d"]])
    assert [c for c, _ in out] == ["a", "c", "b", "d"]


def test_empty() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_duplicate_in_one_list_counted_once() -> None:
    out = dict(reciprocal_rank_fusion([["a", "a"]]))
    assert out["a"] == pytest.approx(1 / 61)
