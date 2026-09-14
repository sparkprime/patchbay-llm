"""The REPL theatre (DESIGN2 §5.2): synchronous turn-taking.

Strict alternation, no preemption, no inbox, no cancellation.  The loop is the
one in §5.2, verbatim::

    while True:
        state = integrate(state, human.read_line(state))
        while turn_at(state) == "llm:main":
            async for event in llm.act(state):
                print_partial(event)
                if event.complete:
                    state = integrate(state, event)

No asynchronous machinery at all -- this is a strict subset of the ACP
theatre, not a different consumer of the same one (DESIGN2 §5.2).
``acp_theatre`` and ``batch_theatre`` are future siblings, not replacements:
each of DESIGN2 §5.1's theatre kinds is a permanent composition rule over the
same participants and the same journal, none of them grows into another.

``default_broadcast`` prints only the new tail of cumulative text per event
id -- accumulate yields cumulative partials (each a complete replacement for
the one before it), so the broadcast tracks how much of each id's text it has
already printed and emits only the difference.  ``exit``, ``quit`` and EOF end
the loop cleanly.
"""

from typing import Awaitable, Callable

from patchbay_llm.conversation import (
    Conversation,
    append_only,
    turn_at,
)
from patchbay_llm.events import Event, Media, Thought
from patchbay_llm.participants.human import HumanParticipant
from patchbay_llm.participants.llm.participant import LlmParticipant

__all__ = ["run_repl", "default_broadcast", "Broadcast"]

BroadcastFn = Callable[[Event], Awaitable[None]]
Broadcast = Callable[[], BroadcastFn]


def default_broadcast() -> BroadcastFn:
    """Print only the new tail of cumulative text per event id.

    Accumulate yields cumulative partials: each is a complete replacement for
    the one before it, not a delta to be applied.  So the broadcast remembers
    the last text it saw for each id and emits only the suffix.  A completed
    message gets a trailing newline.  Thought events are wrapped in
    ``<thinking>`` / ``</thinking>`` as they stream.

    The close tag is emitted when the stream transitions away from a thought
    event (a message event arriving mid-stream), not when the thought's own
    ``complete=True`` promotion arrives -- that promotion comes at ``Finish``,
    after the message text has already streamed, so waiting for it would nest
    the response inside the thinking tag.
    """
    seen: dict[str, str] = {}
    thinking_open = False

    def _close_thinking() -> None:
        nonlocal thinking_open
        if thinking_open:
            print("</thinking>")
            thinking_open = False

    async def _emit(event: Event) -> None:
        nonlocal thinking_open
        if not event.content:
            return
        block = event.content[0]
        if event.kind == "message" and isinstance(block, Media):
            if block.mime != "text/plain":
                return
            _close_thinking()
            text = block.data.decode("utf-8", errors="replace")
            prev = seen.get(event.id, "")
            tail = text[len(prev) :] if text.startswith(prev) else text
            if tail:
                print(tail, end="", flush=True)
            seen[event.id] = text
            if event.complete:
                print()
        elif event.kind == "thought" and isinstance(block, Thought):
            text = block.text
            prev = seen.get(event.id, "")
            if not prev:
                print("<thinking>", end="", flush=True)
                thinking_open = True
            tail = text[len(prev) :] if text.startswith(prev) else text
            if tail:
                print(tail, end="", flush=True)
            seen[event.id] = text
            if event.complete:
                _close_thinking()

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
            async for event in llm.act(state):
                await emit(event)
                if event.complete:
                    state = append_only(state, event)
