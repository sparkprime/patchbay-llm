"""The stream: deltas, stop reason, usage, and the terminal Finish.

The stream yields **increments** (not cumulative values); ``accumulate`` is the
named function between increments and the journal's cumulative partial events,
and stays participant policy.  ``slot`` is the identity of one logical block of
the reply -- one thinking block, one run of text, one tool call -- so that
``accumulate`` has a stable key to map onto one Event id.

The stream promises exactly two ordering guarantees:

1. deltas sharing a ``slot`` arrive in order;
2. ``Finish`` is the last item of a stream that ran to completion.

It deliberately does *not* promise that blocks do not interleave.
"""

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum, auto
from typing import Union

__all__ = [
    "TextDelta",
    "ThoughtDelta",
    "CallDelta",
    "Stop",
    "Usage",
    "Finish",
    "Delta",
]


@dataclass(frozen=True)
class TextDelta:
    """An increment of generated text for block ``slot``."""

    slot: int
    text: str


@dataclass(frozen=True)
class ThoughtDelta:
    """An increment of reasoning text for block ``slot``.

    ``signature`` arrives on the last delta of the slot.
    """

    slot: int
    text: str = ""
    signature: str | None = None


@dataclass(frozen=True)
class CallDelta:
    """An increment of a tool-call's arguments for block ``slot``.

    ``id`` and ``tool`` arrive on the first delta of the slot only.  ``args`` is
    a JSON *fragment*, not valid JSON alone -- it must not be parsed mid-stream.
    """

    slot: int
    id: str | None = None
    tool: str | None = None
    args: str = ""


class Stop(Enum):
    """Why the model stopped."""

    END = auto()  # the model finished on its own
    TOOLS = auto()  # it stopped in order to call tools
    LENGTH = auto()  # it hit max_output -- the reply is truncated
    SEQUENCE = auto()  # a stop sequence fired
    FILTER = auto()  # provider-side content filtering


@dataclass(frozen=True)
class Usage:
    """Token counts, normalised so the four input counts are disjoint.

    ``total prompt tokens == input + cache_read + cache_write``.

    ``output`` is the total generated tokens and ``thought`` is the portion of
    it that was reasoning -- a subset, not an addend.

    ``billed`` is what the provider says it charged (ground truth), when it
    says.  ``estimated`` is true if the adapter had to reconstruct the counts.
    """

    model: str
    input: int
    cache_read: int
    cache_write: int
    output: int
    thought: int
    billed: Decimal | None = None
    estimated: bool = False


@dataclass(frozen=True)
class Finish:
    """The terminal item of a stream that ran to completion.

    Carries both the stop reason and the usage; combining them enforces the
    invariant structurally -- you cannot have a completed call with no counts,
    and you cannot get counts twice.
    """

    stop: Stop
    usage: Usage
    detail: str = ""


Delta = Union[TextDelta, ThoughtDelta, CallDelta, Finish]
