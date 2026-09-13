"""Model facts, prices and token counts.

``Catalogue`` is split from ``InferEngine`` because ``cost_of(state)`` is a
fold over usage events and must work with no engine in the process at all;
``render`` and ``compare`` run offline and need ``window`` and ``measure``;
and ``Catalogue`` has no network dependency.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from typing import Mapping

from patchbay_llm.infer_engine.delta import Usage
from patchbay_llm.infer_engine.prompt import Prompt
from patchbay_llm.infer_engine.request import Effort

__all__ = ["Rate", "Facts", "Catalogue"]


@dataclass(frozen=True)
class Rate:
    """Currency units per token, one entry per category."""

    input: Decimal
    cache_read: Decimal
    cache_write: Decimal
    output: Decimal


@dataclass(frozen=True)
class Facts:
    """Quantities that are both reliable and load-bearing.  No capability booleans.

    ``thinking`` maps each :class:`Effort` to the token budget the adapter will
    actually request (0 means unsupported -> effort-style provider).
    ``rates`` is ``(threshold, Rate)`` pairs ascending by threshold; the
    threshold is the total prompt-token count above which the rate applies.
    """

    model: str
    window: int
    max_output: int
    breakpoints: int
    breakpoint_min: int
    thinking: Mapping[Effort, int]
    rates: tuple[tuple[int, Rate], ...]


class Catalogue(ABC):
    """A second, separate interface -- no network, no inference."""

    @abstractmethod
    def facts(self, model: str) -> Facts:
        """Return the reliable, load-bearing quantities for ``model``."""

    @abstractmethod
    def price(self, usage: Usage) -> Decimal:
        """Return the computed cost for a usage record."""

    @abstractmethod
    def measure(self, prompt: Prompt, model: str) -> int:
        """Return an approximate token count for ``prompt`` under ``model``."""
