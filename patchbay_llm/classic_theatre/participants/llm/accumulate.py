"""``accumulate``: provider deltas -> LiveUpdate / Event.

``infer_engine`` yields ``Delta`` increments; the journal stores ``Event`` s.
This module is the bridge -- it assigns ids, accumulates per-slot deltas,
converts each delta to a :class:`~patchbay_llm.live.LiveUpdate` (via
:func:`~patchbay_llm.infer_engine.convert.to_live_update`), and yields:

- :class:`LiveUpdate` -- one per delta as it arrives, carrying the slot id
  and the per-variant increment fields.  Consumers that want live text (a
  REPL printer, an ACP forwarder) pattern-match on the variants in the same
  ``async for`` loop.
- :class:`Event` -- the final, reified event per slot, yielded on ``Finish``
  (one per slot), plus a terminal ``usage`` event.

``author`` is hardcoded to ``ASSISTANT`` -- this module lives inside the
classic theatre's LLM participant and only ever runs on the assistant's
behalf.

The :class:`LiveUpdate` vocabulary is defined in ``patchbay_llm.live`` -- a
fourth vocabulary, beside inference/journal/ACP, owned by neither
``infer_engine`` nor any theatre.  The Delta → LiveUpdate translation itself
lives in ``infer_engine.convert`` alongside the ContentBlock ↔ Part
translations, keeping all vocabulary seams in one place.
"""

from dataclasses import dataclass
from time import time
from typing import AsyncGenerator, AsyncIterator, Sequence

from patchbay_llm.classic_theatre.conversation import ASSISTANT
from patchbay_llm.events import (
    ContentBlock,
    Event,
    EventId,
    Media,
    Thought,
    ToolCall,
    ToolResult,
    new_id,
)
from patchbay_llm.infer_engine.convert import to_live_update
from patchbay_llm.infer_engine.delta import (
    Delta,
    Finish,
    TextDelta,
    ThoughtDelta,
    Usage,
)
from patchbay_llm.live import (
    LiveUpdate,
    MessageChunk,
    ThoughtChunk,
    ToolCallChunk,
    ToolResultChunk,
)

__all__ = [
    "accumulate",
    "reify_block",
]


# ── The free-standing join function ─────────────────────────────────────────


def reify_block(kind: str, updates: Sequence[LiveUpdate]) -> ContentBlock:
    """Join ordered :class:`LiveUpdate` increments for one slot into a content block.

    Pure function of ``(kind, ordered updates) -> ContentBlock``, callable from
    both live accumulation and a recovery routine running in a different
    process.
    """
    if kind == "message":
        text = "".join(u.text for u in updates if isinstance(u, MessageChunk))
        return Media("text/plain", text.encode())

    if kind == "thought":
        text = "".join(u.text for u in updates if isinstance(u, ThoughtChunk))
        signature: str | None = None
        for u in updates:
            if isinstance(u, ThoughtChunk) and u.signature is not None:
                signature = u.signature
        return Thought(text, signature)

    if kind == "tool_call":
        call_id = ""
        tool = ""
        args = ""
        for u in updates:
            if isinstance(u, ToolCallChunk):
                if u.call_id is not None:
                    call_id = u.call_id
                if u.tool is not None:
                    tool = u.tool
                if u.args:
                    args += u.args
        return ToolCall(call_id, tool, args)

    if kind == "tool_result":
        call_id = ""
        text = ""
        failed = False
        for u in updates:
            if isinstance(u, ToolResultChunk):
                if u.call_id:
                    call_id = u.call_id
                if u.text:
                    text += u.text
                if u.failed is not None:
                    failed = u.failed
        return ToolResult(call_id, (Media("text/plain", text.encode()),), failed)

    raise ValueError(f"unknown slot kind: {kind!r}")


# ── accumulate ─────────────────────────────────────────────────────────────


@dataclass
class _Slot:
    """Internal per-slot buffer: id, kind, accumulated live updates."""

    id: EventId
    kind: str
    updates: list[LiveUpdate]


def _slot_kind(delta: Delta) -> str:
    if isinstance(delta, TextDelta):
        return "message"
    if isinstance(delta, ThoughtDelta):
        return "thought"
    return "tool_call"


def _usage_event(usage: Usage) -> Event:
    return Event(
        id=new_id(),
        ts=time(),
        kind="usage",
        meta={
            "author": ASSISTANT,
            "model": usage.model,
            "input": usage.input,
            "cache_read": usage.cache_read,
            "cache_write": usage.cache_write,
            "output": usage.output,
            "thought": usage.thought,
            "billed": usage.billed,
            "estimated": usage.estimated,
        },
        complete=True,
    )


async def accumulate(
    stream: AsyncIterator[Delta],
) -> AsyncGenerator[Event | LiveUpdate, None]:
    """Turn one inference-delta stream into journal items.

    Yields :class:`LiveUpdate` per delta as it arrives (live increments for
    any consumer that wants them) and :class:`Event` per slot on ``Finish``
    (the final, reified record the theatre integrates into state), plus a
    terminal ``usage`` event.

    A consumer that only wants finals::

        async for item in accumulate(stream):
            if isinstance(item, Event):
                state = integrate(state, item)

    A consumer that wants live text::

        async for item in accumulate(stream):
            match item:
                case MessageChunk(text=t):
                    print(t, end="", flush=True)

    Both happen in the same ``async for`` loop -- no queues, no background
    tasks, no subscriber machinery.  :class:`LiveUpdate` is a neutral,
    UI-facing vocabulary (see ``patchbay_llm.live``), so consumers never need
    ``infer_engine`` types.
    """
    slots: dict[int, _Slot] = {}

    async for delta in stream:
        if isinstance(delta, Finish):
            for slot in slots.values():
                block = reify_block(slot.kind, slot.updates)
                yield Event(
                    id=slot.id,
                    ts=time(),
                    kind=slot.kind,
                    content=(block,),
                    meta={
                        "author": ASSISTANT,
                        "stop": delta.stop.name,
                        "detail": delta.detail,
                    },
                    complete=True,
                )
            yield _usage_event(delta.usage)
            return

        slot = slots.get(delta.slot)
        if slot is None:
            slot = _Slot(id=new_id(), kind=_slot_kind(delta), updates=[])
            slots[delta.slot] = slot

        update = to_live_update(slot.id, delta)
        slot.updates.append(update)
        yield update

    raise RuntimeError("delta stream ended without a Finish")
