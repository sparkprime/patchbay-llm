"""``accumulate``: provider deltas -> cumulative partial journal Events.

``StreamEvent``/``Delta`` and ``Event`` are one step apart, and it is tempting
to delete the gap by having the model layer yield partial events directly. It
should not: assigning ids and accumulating cumulative content is *participant
policy*, whereas ``infer_engine`` exists purely to normalise provider chunk
shapes into ``Delta`` -- keeping this out of it is what makes swapping the
model layer a one-file change (DESIGN2 §6.1).

One journal :class:`~patchbay_llm.events.Event` per ``slot``: a run of text
becomes a ``message`` event, a thinking block a ``thought`` event, a tool
call a ``tool_call`` event. Every delta yields a fresh *cumulative* partial
for its slot (``complete=False``) -- not a delta to be applied, a complete
replacement for the one before it (DESIGN2 §3 "Partial events, merged by
id"). The terminal ``Finish`` promotes every still-open slot to
``complete=True`` and yields one more ``usage`` event.

``CallDelta.args`` is a raw JSON *fragment* and is never parsed here --
``ToolCall.args`` stays a string on the journal until a *complete* event is
handed to ``participants.llm.convert.to_part``. That is the type-level
guarantee that partial tool-call arguments are never executable (DESIGN2 §3).

Interruption needs no special handling. If the caller stops iterating this
generator before ``Finish`` arrives, every slot's last yielded partial is
already the honest, ``complete=False`` record of what was seen -- there is
nothing left to flush and nothing to clean up (DESIGN2 "Surrender is free").
"""

from time import time
from typing import AsyncIterator

from patchbay_llm.events import (
    ContentBlock,
    Event,
    EventId,
    Media,
    Thought,
    ToolCall,
    new_id,
)
from patchbay_llm.infer_engine.delta import (
    Delta,
    Finish,
    TextDelta,
    ThoughtDelta,
)

__all__ = ["accumulate"]


class _Slot:
    """Per-slot accumulation state for one logical block of the reply."""

    __slots__ = (
        "id",
        "kind",
        "text",
        "thought",
        "signature",
        "call_id",
        "tool",
        "args",
    )

    def __init__(self, kind: str) -> None:
        self.id: EventId = new_id()
        self.kind = kind
        self.text: list[str] = []
        self.thought: list[str] = []
        self.signature: str | None = None
        self.call_id: str | None = None
        self.tool: str | None = None
        self.args: list[str] = []

    def block(self) -> ContentBlock:
        """The cumulative content block for this slot, as of right now."""
        if self.kind == "message":
            return Media("text/plain", "".join(self.text).encode())
        if self.kind == "thought":
            return Thought("".join(self.thought), self.signature)
        return ToolCall(self.call_id or "", self.tool or "", "".join(self.args))


def _event(slot: _Slot, author: str, *, complete: bool) -> Event:
    return Event(
        id=slot.id,
        ts=time(),
        kind=slot.kind,
        content=(slot.block(),),
        meta={"author": author},
        complete=complete,
    )


def _usage_event(finish: Finish, author: str) -> Event:
    u = finish.usage
    return Event(
        id=new_id(),
        ts=time(),
        kind="usage",
        meta={
            "author": author,
            "model": u.model,
            "input": u.input,
            "cache_read": u.cache_read,
            "cache_write": u.cache_write,
            "output": u.output,
            "thought": u.thought,
            "billed": u.billed,
            "estimated": u.estimated,
            "stop": finish.stop.name,
            "detail": finish.detail,
        },
        complete=True,
    )


async def accumulate(stream: AsyncIterator[Delta], author: str) -> AsyncIterator[Event]:
    """Turn one participant's inference-delta stream into journal Events.

    ``author`` becomes ``meta.author`` on every emitted event (DESIGN2's
    journal vocabulary: message/thought/tool_call are authored). Passed in
    rather than read from anywhere ambient, so this function is testable with
    nothing but a synthetic ``Delta`` sequence -- no participant, no engine,
    no journal.
    """
    slots: dict[int, _Slot] = {}

    async for delta in stream:
        if isinstance(delta, Finish):
            for slot in slots.values():
                yield _event(slot, author, complete=True)
            yield _usage_event(delta, author)
            return

        slot: _Slot | None
        if isinstance(delta, TextDelta):
            slot = slots.get(delta.slot)
            if slot is None:
                slot = _Slot("message")
                slots[delta.slot] = slot
            slot.text.append(delta.text)
        elif isinstance(delta, ThoughtDelta):
            slot = slots.get(delta.slot)
            if slot is None:
                slot = _Slot("thought")
                slots[delta.slot] = slot
            if delta.text:
                slot.thought.append(delta.text)
            if delta.signature is not None:
                slot.signature = delta.signature
        else:  # CallDelta -- the only remaining member of the Delta union
            slot = slots.get(delta.slot)
            if slot is None:
                slot = _Slot("tool_call")
                slots[delta.slot] = slot
            if delta.id is not None:
                slot.call_id = delta.id
            if delta.tool is not None:
                slot.tool = delta.tool
            if delta.args:
                slot.args.append(delta.args)

        yield _event(slot, author, complete=False)

    # The stream exhausted with no Finish -- infer_engine's own contract (§5)
    # forbids this (it must raise instead), so reaching here is a bug one
    # layer down, not something to paper over here.
    raise RuntimeError("delta stream ended without a Finish")
