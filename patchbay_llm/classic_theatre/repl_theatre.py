"""The REPL theatre (DESIGN2 §5.2): synchronous turn-taking.

Strict alternation, no preemption, no inbox, no cancellation.  The loop is the
one in §5.2:

    while True:
        text = source()                           # human turn (inlined)
        if text is None or text in {exit, quit}:
            return
        state = integrate(state, message(HUMAN, text))
        state = integrate(state, turn_end(HUMAN))
        while turn_at(state) == ASSISTANT:
            stream = accumulate(engine.run(Request(model, prompt, knobs)))
            async for item in stream:
                if isinstance(item, LiveUpdate):
                    sink(live text)               # synchronous, same loop
                else:
                    state = integrate(state, item)

Both sides are inlined directly into the loop: the human turn is a single
blocking read (no inference, no state consultation), and the assistant turn is
a single inference call (render state -> request -> accumulate deltas -> yield
events).  Neither needed the indirection of a participant class.

``run_repl`` owns the rendering bookkeeping — thinking-tag wrapping, trailing
newlines on completed messages, and de-duplication of text already streamed via
:class:`LiveUpdate` against the final :class:`Event` — so that callers provide
only raw I/O: a ``source`` that returns one line at a time (line-oriented,
blocking) and a ``sink`` that prints a text chunk (arbitrary fragment, not
necessarily a whole line).  The example in ``examples/`` is a few lines of
``input()`` / ``print()`` and nothing else.

``accumulate`` yields :class:`LiveUpdate` per delta (live increments, processed
synchronously via pattern match on the variant) and :class:`Event` per slot on
``Finish`` (the final record the theatre integrates).  No queues, no background
tasks, no subscriber machinery — everything flows through one ``async for``
loop.

This module owns both directions of the journal↔inference seam for the classic
theatre: :func:`render` folds journal state into a prompt (journal → inference),
and :func:`accumulate` folds a delta stream into journal items (inference →
journal) — the two are opposites, and keeping them together makes that visible.
The Delta → :class:`LiveUpdate` translation lives in
``infer_engine.convert`` and the :class:`LiveUpdate` → :class:`ContentBlock`
join in ``patchbay_llm.reify``; ``accumulate`` is the one place that orchestrates
both into :class:`Event` s.
"""

from dataclasses import dataclass
from time import time
from typing import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Sequence,
)

from patchbay_llm.classic_theatre.conversation import (
    ASSISTANT,
    HUMAN,
    Conversation,
    append_only,
    message,
    turn_at,
    turn_end,
)
from patchbay_llm.events import Event, EventId, new_id
from patchbay_llm.infer_engine.convert import to_live_update, to_part
from patchbay_llm.infer_engine.delta import (
    Delta,
    Finish,
    TextDelta,
    ThoughtDelta,
)
from patchbay_llm.infer_engine.engine import InferEngine
from patchbay_llm.infer_engine.prompt import Message, Prompt, Role
from patchbay_llm.infer_engine.request import Knobs, Request
from patchbay_llm.live import LiveUpdate, MessageChunk, ThoughtChunk
from patchbay_llm.reify import reify_block

__all__ = [
    "run_repl",
    "render",
    "accumulate",
    "Source",
    "Sink",
]

Source = Callable[[], Awaitable[str | None]]

Sink = Callable[[str], None]


# ── render: the state -> Prompt fold ────────────────────────────────────────

_CONTENT_KINDS = ("message", "thought", "tool_call", "tool_result")


def _role(author: str) -> Role:
    assert author in (HUMAN, ASSISTANT)
    return "llm" if author == ASSISTANT else "user"


def render(state: Sequence[Event]) -> Prompt:
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
    updates: list[LiveUpdate]


def _slot_kind(delta: Delta) -> str:
    if isinstance(delta, TextDelta):
        return "message"
    if isinstance(delta, ThoughtDelta):
        return "thought"
    return "tool_call"


async def accumulate(
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
            slot = _Slot(id=new_id(), kind=_slot_kind(delta), updates=[])
            slots[delta.slot] = slot

        update = to_live_update(slot.id, delta)
        slot.updates.append(update)
        yield update

    raise RuntimeError("delta stream ended without a Finish")


async def run_repl(
    source: Source,
    sink: Sink,
    engine: InferEngine,
    model: str,
    knobs: Knobs = Knobs(),
) -> None:
    """Run a two-party REPL conversation to EOF or ``exit``/``quit``.

    ``source`` is an awaitable that returns one line of text, or ``None`` at
    EOF.  ``sink`` is a plain ``print``-like callable that receives each text
    chunk as it streams — never a whole journal item, just ``str``.

    The theatre owns the rendering bookkeeping so that callers need only raw
    I/O: thinking events are wrapped in ``<thinking>`` / ``</thinking>`` as they
    stream, a completed LLM message gets a trailing newline, and text already
    printed via :class:`LiveUpdate` deltas is not re-printed when the final
    :class:`Event` arrives.  Tool-call increments are not rendered live (the
    model's argument stream is not useful to watch character-by-character; the
    final ``tool_call`` Event is what the user sees).  ``exit``, ``quit`` and
    EOF end the loop cleanly.  Empty input is re-read until a non-empty line
    arrives.
    """
    state: Conversation = ()
    seen_ids: set[str] = set()
    thinking_open = False

    while True:
        # ── Human turn ─────────────────────────────────────────────────
        while True:
            text = await source()
            if text is None or text.strip() in ("exit", "quit"):
                return
            if not text.strip():
                continue
            state = append_only(state, message(HUMAN, text))
            state = append_only(state, turn_end(HUMAN, "relinquished"))
            break

        # ── Assistant turn ─────────────────────────────────────────────
        while turn_at(state) == ASSISTANT:
            request = Request(model=model, prompt=render(state), knobs=knobs)
            async for item in accumulate(engine.run(request)):
                if isinstance(item, Event):
                    if item.id in seen_ids:
                        if item.kind == "thought" and thinking_open:
                            sink("</thinking>")
                            thinking_open = False
                        elif item.kind == "message" and item.complete:
                            if thinking_open:
                                sink("</thinking>")
                                thinking_open = False
                            sink("\n")
                    state = append_only(state, item)
                elif isinstance(item, (MessageChunk, ThoughtChunk)):
                    if not item.text:
                        continue
                    seen_ids.add(item.id)
                    if isinstance(item, ThoughtChunk):
                        if not thinking_open:
                            sink("<thinking>")
                            thinking_open = True
                    elif thinking_open:
                        sink("</thinking>")
                        thinking_open = False
                    sink(item.text)
            state = append_only(state, turn_end(ASSISTANT, "relinquished"))
