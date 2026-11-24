"""Parse inline [n] citation markers and validate them against the sources."""
import re
from collections.abc import Sequence
from dataclasses import dataclass

from app.retrieval import Candidate

IDK_ANSWER = "I don't know based on the provided documents"
SNIPPET_CHARS = 200

_FENCED = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`\n]*`")
# [2] or [1, 3]; not followed by "(" (markdown link).
_MARKER = re.compile(r"\[\s*([0-9]+(?:\s*,\s*[0-9]+)*)\s*\](?!\()")


@dataclass
class Citation:
    """A cited marker resolved (or not) to a retrieved source."""

    marker: int
    filename: str | None
    page: int | None
    chunk_id: str | None
    snippet: str | None
    invalid: bool = False


def is_idk(answer: str) -> bool:
    """True only if the whole reply is the 'I don't know' sentence (normalised)."""
    strip = "*_`\"'\u201c\u201d\u2018 \t\r\n"
    norm = answer.replace("\u2019", "'").strip().strip(strip)
    norm = norm.rstrip(".").strip(strip)
    return norm.lower() == IDK_ANSWER.lower()


def parse_markers(answer: str) -> list[int]:
    """Citation numbers in first-appearance order, deduplicated."""
    text = _INLINE_CODE.sub(" ", _FENCED.sub(" ", answer))
    seen: dict[int, None] = {}
    for m in _MARKER.finditer(text):
        for num in m.group(1).split(","):
            seen.setdefault(int(num), None)
    return list(seen)


def make_snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    """Whitespace-normalised excerpt of at most `limit` characters."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 3].rstrip() + "..."


def build_citations(answer: str, sources: Sequence[Candidate]) -> list[Citation]:
    """Resolve markers; sources[i] is marker i+1. Unknown markers are invalid."""
    if is_idk(answer):
        return []
    out: list[Citation] = []
    for n in parse_markers(answer):
        if 1 <= n <= len(sources):
            c = sources[n - 1]
            out.append(Citation(n, c.filename, c.page, c.chunk_id, make_snippet(c.text)))
        else:
            out.append(Citation(n, None, None, None, None, invalid=True))
    return out
