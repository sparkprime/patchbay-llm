"""The REPL theatre (DESIGN2 §5.2): synchronous turn-taking.

Strict alternation, no preemption, no inbox, no cancellation.  The loop is the
one in §5.2, adapted for the ``PartialEvent`` model (durable_partials.md):

    while True:
        state = integrate(state, human.read_line(state))
        while turn_at(state) == "llm:main":
            async for item in llm.act(state):
                if isinstance(item, PartialEvent):
                    spawn subscriber task   # live text
                else:
                    drain subscriber tasks  # no race with finals
                    state = integrate(state, item)

The human side yields complete :class:`Event` s only (no PartialEvents), so it
is unchanged.  The LLM side yields ``Event | PartialEvent``: a
:class:`PartialEvent` handle once per new slot, then final :class:`Event` s on
``Finish``.  The theatre spawns a background task per PartialEvent to subscribe
and print deltas live, and drains all active subscriber tasks before
processing a final Event -- ensuring no text is lost to a race between the
subscriber and the final Event's newline.

``default_broadcast`` prints the actual increment per delta -- no diffing
against a previous cumulative snapshot (durable_partials.md: the whole point
of ``subscribe()``).  A completed LLM message gets a trailing newline.  Thought
events are wrapped in ``<thinking>`` / ``</thinking>`` as they stream.
``exit``, ``quit`` and EOF end the loop cleanly.
"""

import asyncio
from typing import Any, Callable, Coroutine, Union

from patchbay_llm.conversation import (
    Conversation,
    append_only,
    turn_at,
)
from patchbay_llm.events import Event, Media
from patchbay_llm.infer_engine.delta import TextDelta, ThoughtDelta
from patchbay_llm.participants.human import HumanParticipant
from patchbay_llm.participants.llm.accumulate import PartialEvent
from patchbay_llm.participants.llm.participant import LlmParticipant

__all__ = ["run_repl", "default_broadcast", "Broadcast", "Item"]

Item = Union[Event, PartialEvent]

BroadcastFn = Callable[[Item], Coroutine[Any, Any, None]]
Broadcast = Callable[[], BroadcastFn]


def default_broadcast() -> BroadcastFn:
    """Print live text as it arrives, and final messages with newlines.

    Under the PartialEvent model, LLM text is printed incrementally via
    ``subscribe()`` -- each delta is the actual increment, no diffing needed
    (durable_partials.md: the whole point of subscribe()).  Human messages
    (which arrive as complete Events, not PartialEvents) are printed in full.
    A completed LLM message gets a trailing newline.  Thought events are
    wrapped in ``<thinking>`` / ``</thinking>`` as they stream.

    The close tag is emitted when the thought's subscription ends (i.e. when
    ``reify()`` is called), not when the final ``complete=True`` Event arrives
    -- that promotion comes at ``Finish``, after the message text may have
    already streamed, so waiting for it would nest the response inside the
    thinking tag.
    """

    partial_ids: set[str] = set()
    thinking_open = False

    def _close_thinking() -> None:
        nonlocal thinking_open
        if thinking_open:
            print("</thinking>")
            thinking_open = False

    async def _emit(item: Item) -> None:
        nonlocal thinking_open
        if isinstance(item, PartialEvent):
            partial_ids.add(item.id)
            async for delta in item.subscribe():
                if isinstance(delta, TextDelta) and item.kind == "message":
                    _close_thinking()
                    print(delta.text, end="", flush=True)
                elif isinstance(delta, ThoughtDelta) and item.kind == "thought":
                    if delta.text:
                        if not thinking_open:
                            print("<thinking>", end="", flush=True)
                            thinking_open = True
                        print(delta.text, end="", flush=True)
            # subscription ended — reify was called
            if item.kind == "thought":
                _close_thinking()
        else:
            event = item
            if event.id in partial_ids:
                # Text already printed via subscription; just finalize.
                if event.kind == "message" and event.complete:
                    _close_thinking()
                    print()
            else:
                # Not from a PartialEvent (human message, turn_end, usage).
                if event.kind == "message":
                    block = event.content[0] if event.content else None
                    if isinstance(block, Media) and block.mime == "text/plain":
                        _close_thinking()
                        print(block.data.decode("utf-8", errors="replace"))

    return _emit


def _message_text(event: Event) -> str:
    if not event.content:
        return ""
    block = event.content[0]
    if isinstance(block, Media) and block.mime == "text/plain":
        return block.data.decode("utf-8", errors="replace")
    return ""


async def run_repl(
    human: HumanParticipant,
    llm: LlmParticipant,
    broadcast: Broadcast = default_broadcast,
) -> None:
    """Run a two-party REPL conversation to EOF or ``exit``/``quit``."""
    participants = (human.me, llm.me)
    state: Conversation = ()
    emit = broadcast()
    while True:
        got_message = False
        async for event in human.act(state):
            if event.kind == "message":
                got_message = True
                if _message_text(event).strip() in ("exit", "quit"):
                    return
            await emit(event)
            if event.complete:
                state = append_only(state, event)
        if not got_message:
            return
        while turn_at(state, participants) == llm.me:
            live: set[asyncio.Task[None]] = set()
            async for item in llm.act(state):
                if isinstance(item, PartialEvent):
                    task = asyncio.create_task(emit(item))
                    live.add(task)
                    task.add_done_callback(live.discard)
                else:
                    if live:
                        await asyncio.gather(*live)
                        live.clear()
                    await emit(item)
                    if item.complete:
                        state = append_only(state, item)
            if live:
                await asyncio.gather(*live)
                live.clear()
