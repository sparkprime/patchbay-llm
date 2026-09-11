"""The InferEngine protocol.

A plain function returning an async iterator -- not ``async def run`` -- so the
call site is ``async for delta in engine.run(request)`` with no extra ``await``.
An async generator satisfies this and gives ``aclose()`` for free.

See INFERENCE.md §7.
"""

from __future__ import annotations

from typing import AsyncIterator, Protocol

from patchbay_llm.infer_engine.delta import Delta
from patchbay_llm.infer_engine.request import Request

__all__ = ["InferEngine"]


class InferEngine(Protocol):
    """One call. Everything patchbay-llm asks of an LLM goes through it.

    Reentrant: one engine serves many concurrent calls and ``run`` holds no
    shared mutable state.
    """

    def run(self, request: Request) -> AsyncIterator[Delta]:
        """Return an async iterator of inference deltas for the request."""
        ...  # pylint: disable=unnecessary-ellipsis
