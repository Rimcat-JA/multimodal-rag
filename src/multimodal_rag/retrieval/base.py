"""The interface every retrieval method implements (A, B1, B2, C).

The evaluation code only talks to this interface, so it never needs to know
how a method works inside. Every method returns whole pages: methods that
index smaller units (e.g. one caption per figure in B) collapse them to pages
before returning, keeping each page's best score.
"""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class PageHit:
    page_id: str  # same ids as data/pages.jsonl and eval/questions.csv
    score: float  # higher is better; only comparable within one method


class Retriever(Protocol):
    def index(self, pages: list[dict]) -> None:
        """Build the index from page records (one dict per line of data/pages.jsonl)."""
        ...

    def search(self, query: str, k: int) -> list[PageHit]:
        """Return at most k pages, best first, with no page twice."""
        ...
