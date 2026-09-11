"""Test helpers shipped with the API.

``check_stream`` verifies the two ordering guarantees the stream promises
(INFERENCE.md §4):

1. deltas sharing a ``slot`` arrive in order;
2. ``Finish`` is the last item of a stream that ran to completion.

It deliberately does *not* promise that blocks do not interleave.
"""

from __future__ import annotations

from typing import Iterable

from patchbay_llm.infer_engine.delta import Delta, Finish

__all__ = ["check_stream"]


def check_stream(deltas: Iterable[Delta]) -> None:
    """Assert the stream ordering contract.  Raises ``AssertionError`` on violation."""
    saw_finish = False
    finish_count = 0
    for delta in deltas:
        if saw_finish:
            raise AssertionError("delta arrived after Finish")
        if isinstance(delta, Finish):
            finish_count += 1
            saw_finish = True
    if finish_count > 1:
        raise AssertionError(f"stream had {finish_count} Finish items; expected one")
