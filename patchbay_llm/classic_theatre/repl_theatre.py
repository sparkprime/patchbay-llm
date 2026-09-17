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
            stream = accumulate(engine.run(Request(model, prompt, knobs)), sink)
            async for item in stream:
                if isinstance(item, LiveUpdate):
                    print live text                # synchronous, same loop
                else:
                    state = integrate(state, item)

Both sides are inlined directly into the loop: the human turn is a single
blocking read (no inference, no state consultation), and the assistant turn is
a single inference call (render state -> request -> accumulate deltas -> yield
events).  Neither needed the indirection of a participant class.

``accumulate`` yields :class:`LiveUpdate` per delta (live increments, processed
synchronously via pattern match on the variant) and :class:`Event` per slot on
``Finish`` (the final record the theatre integrates).  No queues, no background
tasks, no subscriber machinery — everything flows through one ``async for``
loop.

``default_broadcast`` prints the actual increment per delta — no diffing
against a previous cumulative snapshot.  A completed LLM message gets a
trailing newline.  Thought events are wrapped in ``<thinking>`` /
``</thinking>`` as they stream.  ``exit``, ``quit`` and EOF end the loop
cleanly.
"""

import asyncio
from typing import Awaitable, Callable, Coroutine, Sequence, Union

from patchbay_llm.classic_theatre.conversation import (
    ASSISTANT,
    HUMAN,
    Conversation,
    append_only,
    message,
    turn_at,
    turn_end,
)
from patchbay_llm.classic_theatre.participants.llm.accumulate import (
    accumulate,
)
from patchbay_llm.events import Event, Media
from patchbay_llm.infer_engine.convert import to_part
from patchbay_llm.infer_engine.engine import InferEngine
from patchbay_llm.infer_engine.prompt import Message, Prompt, Role
from patchbay_llm.infer_engine.request import Knobs, Request
from patchbay_llm.live import LiveUpdate, MessageChunk, ThoughtChunk

__all__ = [
    "run_repl",
    "default_broadcast",
    "render",
    "Broadcast",
    "Item",
    "Source",
    "terminal_source",
]

Item = Union[Event, LiveUpdate]

Source = Callable[[], Awaitable[str | None]]

BroadcastFn = Callable[[Item], Coroutine[None, None, None]]
Broadcast = Callable[[], BroadcastFn]


async def append_and_emit(
    state: Conversation, emit: BroadcastFn, event: Event
) -> Conversation:
    """Emit an event and integrate it into state, in one line."""
    await emit(event)
    return append_only(state, event)


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


def default_broadcast() -> BroadcastFn:
    """Print live text as it arrives, and final messages with newlines.

    Each :class:`LiveUpdate` is the actual increment — no diffing, no cumulative
    snapshot.  Human messages (which arrive as complete Events) are printed in
    full.  A completed LLM message gets a trailing newline.  Thought events are
    wrapped in ``<thinking>`` / ``</thinking>`` as they stream.  Tool-call
    increments are not rendered live in the terminal today (the model's
    argument stream is not useful to watch character-by-character; the
    final ``tool_call`` Event is what the user sees).
    """

    seen_ids: set[str] = set()
    thinking_open = False

    def _close_thinking() -> None:
        nonlocal thinking_open
        if thinking_open:
            print("</thinking>")
            thinking_open = False

    async def _emit(item: Item) -> None:
        nonlocal thinking_open
        match item:
            case MessageChunk(id=uid, text=t) if t:
                seen_ids.add(uid)
                _close_thinking()
                print(t, end="", flush=True)
            case ThoughtChunk(id=uid, text=t) if t:
                seen_ids.add(uid)
                if not thinking_open:
                    print("<thinking>", end="", flush=True)
                    thinking_open = True
                print(t, end="", flush=True)
            case Event():
                event = item
                if event.id in seen_ids:
                    # Text already printed via deltas; just finalize.
                    if event.kind == "thought":
                        _close_thinking()
                    if event.kind == "message" and event.complete:
                        _close_thinking()
                        print()
                else:
                    # Not from a LiveUpdate (human message, turn_end, usage).
                    if event.kind == "message":
                        block = event.content[0] if event.content else None
                        if isinstance(block, Media) and block.mime == "text/plain":
                            _close_thinking()
                            print(block.data.decode("utf-8", errors="replace"))

    return _emit


def terminal_source(prompt: str = "> ") -> Source:
    """A ``source`` backed by blocking ``input()``, wrapped in an executor.

    Blocking ``input()`` would freeze the event loop, so it runs in the
    default executor's thread pool and is awaited.  ``EOFError`` (Ctrl-D)
    becomes ``None`` -- the end-of-input signal the theatre reads to exit.
    """

    async def _read() -> str | None:
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(None, lambda: input(prompt))
        except EOFError:
            return None

    return _read


terminal_sink = default_broadcast()


async def run_repl(
    source: Source,
    sink: BroadcastFn,
    engine: InferEngine,
    model: str,
    knobs: Knobs = Knobs(),
) -> None:
    """Run a two-party REPL conversation to EOF or ``exit``/``quit``.

    The human turn is a single blocking read from ``source``.  The assistant
    turn is a single inference call.  Both are inlined — no participant
    classes.  ``source`` returns ``str | None``; ``None`` (EOF) and the
    commands ``exit``/``quit`` end the loop cleanly.  Empty input is re-read
    until a non-empty line arrives.
    """
    state: Conversation = ()
    while True:
        # ── Human turn ─────────────────────────────────────────────────────
        while True:
            text = await source()
            if text is None or text.strip() in ("exit", "quit"):
                return
            if not text.strip():
                continue
            msg = message(HUMAN, text)
            state = await append_and_emit(state, sink, msg)
            state = await append_and_emit(state, sink, turn_end(HUMAN, "relinquished"))
            break

        # ── Assistant turn ─────────────────────────────────────────────────
        while turn_at(state) == ASSISTANT:
            request = Request(model=model, prompt=render(state), knobs=knobs)
            async for item in accumulate(engine.run(request)):
                await sink(item)
                if isinstance(item, Event) and item.complete:
                    state = append_only(state, item)
            state = await append_and_emit(
                state, sink, turn_end(ASSISTANT, "relinquished")
            )
