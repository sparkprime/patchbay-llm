"""The journal↔inference seam for the classic theatre.

This module owns both directions of the seam: :func:`render` folds journal state
into a prompt (journal → inference), and :func:`accumulate` folds a delta stream
into journal items (inference → journal).  They are opposites and live together
to make that relationship visible.

The Delta → :class:`LiveUpdate` translation lives in
``infer_engine.convert`` and the :class:`LiveUpdate` → :class:`ContentBlock`
join in ``patchbay_llm.reify``; ``accumulate`` is the one place that orchestrates
both into :class:`Event` s.
"""

from dataclasses import dataclass, field
from time import time
from typing import AsyncGenerator, AsyncIterator

from patchbay_llm.classic_theatre.conversation import (
    ASSISTANT,
    Conversation,
)
from patchbay_llm.events import Event, EventId, new_id
from patchbay_llm.freeze import freeze_block
from patchbay_llm.infer_engine.convert import to_live_update, to_part
from patchbay_llm.infer_engine.delta import (
    Delta,
    Finish,
    TextDelta,
    ThoughtDelta,
)
from patchbay_llm.infer_engine.prompt import Message, Prompt, Role
from patchbay_llm.live import LiveUpdate

# ── render: the state -> Prompt fold ────────────────────────────────────────

_CONTENT_KINDS = ("message", "thought", "tool_call", "tool_result")


def _role(author: str) -> Role:
    assert author in ("human", "assistant")
    return "llm" if author == "assistant" else "user"


def to_prompt(state: Conversation) -> Prompt:
    """Turn the conversation into a prompt.

    Ignore unrecognised kinds, concatenate repeated roles.
    """
    messages: list[Message] = []
    for e in state:
        if e.kind not in _CONTENT_KINDS:
            continue
        role = _role(e.meta["author"])
        content_as_parts = tuple(to_part(b) for b in e.content)
        if messages and messages[-1].role == role:
            # Ensure roles take turns in the prompt.
            prev = messages[-1]
            messages[-1] = Message(role, prev.parts + content_as_parts)
        else:
            messages.append(Message(role, content_as_parts))
    return Prompt(messages=tuple(messages))


# ── accumulate: the inference -> journal fold (the opposite of render) ──────


@dataclass
class _Slot:
    """Internal per-slot buffer: id, kind, accumulated live updates."""

    id: EventId
    kind: str
    updates: list[LiveUpdate] = field(default_factory=list[LiveUpdate])


def _slot_kind(delta: Delta) -> str:
    if isinstance(delta, TextDelta):
        return "message"
    if isinstance(delta, ThoughtDelta):
        return "thought"
    return "tool_call"


async def to_events_and_live_updates(
    stream: AsyncIterator[Delta],
) -> AsyncGenerator[Event | LiveUpdate, None]:
    """Turn one inference-delta stream into journal items.

    The inverse of :func:`render`: where ``render`` folds journal state into a
    prompt, ``accumulate`` folds an inference-delta stream into journal items.
    It assigns an :class:`EventId` per slot, converts each :class:`Delta` to a
    :class:`LiveUpdate` (via :func:`~patchbay_llm.infer_engine.convert.to_live_update`)
    and yields it, and on :class:`Finish` joins the per-slot increments into the
    one final :class:`Event` per slot (via :func:`patchbay_llm.reify.reify_block`)
    plus a terminal ``usage`` event.  ``author`` is hardcoded to
    :data:`~patchbay_llm.classic_theatre.conversation.ASSISTANT` -- this fold
    only ever runs on the assistant's behalf.

    Yields :class:`LiveUpdate` per delta as it arrives (live increments for any
    consumer that wants them) and :class:`Event` per slot on ``Finish`` (the
    final, reified record the theatre integrates into state), plus a terminal
    ``usage`` event.

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
                block = freeze_block(slot.kind, slot.updates)
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
            usage = delta.usage
            yield Event(
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
            return

        slot = slots.get(delta.slot)
        if slot is None:
            slot = _Slot(id=new_id(), kind=_slot_kind(delta))
            slots[delta.slot] = slot

        update = to_live_update(slot.id, delta)
        slot.updates.append(update)
        yield update

    raise RuntimeError("delta stream ended without a Finish")
