"""Draining, for tests and simple callers.

``collect`` is one helper, clearly labelled convenience -- not a second code
path: there is no non-streaming call to make, and ``collect`` is a thin fold
over the public stream.  ``Reply`` keeps the **blocks**, in order, so that
``as_message()`` is an identity and the tool loop is
``prompt.extend(reply.as_message(), results)`` with no translation step.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, AsyncIterator, Mapping

from patchbay_llm.infer_engine.delta import (
    Delta,
    Finish,
    Stop,
    TextDelta,
    ThoughtDelta,
    Usage,
)
from patchbay_llm.infer_engine.prompt import Call, Media, Message, Part, Thought

__all__ = ["Reply", "collect"]


@dataclass(frozen=True)
class Reply:
    """A completed reply: blocks in order, plus the stop reason and usage."""

    parts: tuple[Part, ...]  # Thought | Media("text/plain") | Call, by slot order
    stop: Stop
    usage: Usage

    @property
    def text(self) -> str:
        """The text parts, joined."""
        out: list[str] = []
        for part in self.parts:
            if isinstance(part, Media) and part.mime == "text/plain":
                out.append(part.data.decode("utf-8", errors="replace"))
        return "".join(out)

    @property
    def calls(self) -> tuple[Call, ...]:
        """The tool-call parts, in order."""
        return tuple(p for p in self.parts if isinstance(p, Call))

    def as_message(self) -> Message:
        """The reply as an llm message -- the identity that closes the loop."""
        return Message("llm", self.parts)


async def collect(stream: AsyncIterator[Delta]) -> Reply:
    """Drain a stream into a :class:`Reply`.

    A ``collect`` that returns at all returns a completed reply; every other
    ending is an exception or a consumer that chose to stop and therefore never
    called ``collect``.
    """

    class _Text:
        buf: list[str] = []

    class _Thought:
        buf: list[str] = []
        signature: str | None = None

    class _Call:
        id: str | None = None
        tool: str | None = None
        args: list[str] = []

    blocks: dict[int, object] = {}
    order: list[int] = []
    stop: Stop | None = None
    usage: Usage | None = None

    async for delta in stream:
        if isinstance(delta, Finish):
            stop = delta.stop
            usage = delta.usage
            continue

        if delta.slot not in blocks:
            if isinstance(delta, TextDelta):
                blocks[delta.slot] = _Text()
            elif isinstance(delta, ThoughtDelta):
                blocks[delta.slot] = _Thought()
            else:
                blocks[delta.slot] = _Call()
            order.append(delta.slot)

        block = blocks[delta.slot]

        if isinstance(delta, TextDelta):
            assert isinstance(block, _Text)
            block.buf.append(delta.text)
        elif isinstance(delta, ThoughtDelta):
            assert isinstance(block, _Thought)
            if delta.text:
                block.buf.append(delta.text)
            if delta.signature is not None:
                block.signature = delta.signature
        else:  # CallDelta
            assert isinstance(block, _Call)
            if delta.id is not None:
                block.id = delta.id
            if delta.tool is not None:
                block.tool = delta.tool
            if delta.args:
                block.args.append(delta.args)

    if stop is None or usage is None:
        raise RuntimeError("stream ended without a Finish")

    parts: list[Part] = []
    for slot in order:
        block = blocks[slot]
        if isinstance(block, _Text):
            parts.append(Media("text/plain", "".join(block.buf).encode("utf-8")))
        elif isinstance(block, _Thought):
            parts.append(Thought("".join(block.buf), block.signature))
        else:
            assert isinstance(block, _Call)
            parsed: Mapping[str, Any] = (
                json.loads("".join(block.args)) if block.args else {}
            )
            parts.append(Call(block.id or "", block.tool or "", parsed))

    return Reply(tuple(parts), stop, usage)
