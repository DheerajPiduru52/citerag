"""Phase 3 tests: citation parsing and validation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.citations import (  # noqa: E402
    IDK_ANSWER, SNIPPET_CHARS, build_citations, is_idk, make_snippet, parse_markers,
)
from app.retrieval import Candidate  # noqa: E402


def cands(n: int = 5) -> list[Candidate]:
    return [
        Candidate(chunk_id=f"c{i}", text=f"text   of\nchunk {i}", doc_id="d",
                  filename=f"f{i}.pdf", page=i)
        for i in range(1, n + 1)
    ]


def test_single_and_order() -> None:
    assert parse_markers("A [3] then B [1].") == [3, 1]


def test_grouped_forms() -> None:
    assert parse_markers("x [1, 3] y [2][4]") == [1, 3, 2, 4]


def test_repeated_deduped_first_appearance() -> None:
    assert parse_markers("[2] a [1] b [2] c [1, 2]") == [2, 1]


def test_non_citations_ignored() -> None:
    text = "see [note] and [link](http://x.com) and `a[1]` and [a1] [1.5]\n```\nb[2]\n```"
    assert parse_markers(text) == []


def test_no_markers() -> None:
    assert build_citations("Plain answer.", cands()) == []


def test_valid_citation_fields() -> None:
    (c,) = build_citations("Yes [2].", cands())
    assert (c.marker, c.filename, c.page, c.chunk_id, c.invalid) == (2, "f2.pdf", 2, "c2", False)
    assert c.snippet == "text of chunk 2"


def test_out_of_range_and_zero_invalid() -> None:
    cs = build_citations("A [7] B [0] C [1].", cands(5))
    assert [c.marker for c in cs] == [7, 0, 1]
    for c in cs[:2]:
        assert c.invalid and c.filename is None and c.page is None
        assert c.chunk_id is None and c.snippet is None
    assert not cs[2].invalid


def test_idk_gives_no_citations() -> None:
    assert build_citations(IDK_ANSWER, cands()) == []
    assert build_citations(IDK_ANSWER + ".", cands()) == []
    assert build_citations("I don’t know based on the provided documents.", cands()) == []
    assert build_citations('**"' + IDK_ANSWER + '."**', cands()) == []
    assert build_citations("`" + IDK_ANSWER.upper() + "`", cands()) == []


def test_idk_prefix_with_content_keeps_citations() -> None:
    text = IDK_ANSWER + ". However [1] says X"
    assert not is_idk(text)
    assert [c.marker for c in build_citations(text, cands())] == [1]


def test_unicode_digits_not_parsed() -> None:
    assert parse_markers("a [١] b [²] c [１] d [1]") == [1]


def test_snippet_length_and_whitespace() -> None:
    long = "word  " * 200
    s = make_snippet(long)
    assert len(s) <= SNIPPET_CHARS and s.endswith("...") and "  " not in s
    assert make_snippet("short\n\ttext") == "short text"
