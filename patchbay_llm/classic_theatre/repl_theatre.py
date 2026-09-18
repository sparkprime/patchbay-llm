"""The REPL theatre: Strict turn taking, no preemption by the user."""

from typing import Awaitable, Callable

from patchbay_llm.classic_theatre.conversation import (
    ASSISTANT,
    HUMAN,
    Conversation,
    append_only,
    message,
    turn_at,
    turn_end,
)
from patchbay_llm.classic_theatre.seam import to_events_and_live_updates, to_prompt
from patchbay_llm.events import Event
from patchbay_llm.infer_engine.engine import InferEngine
from patchbay_llm.infer_engine.request import Knobs, Request
from patchbay_llm.live import MessageUpdate, ThoughtUpdate

__all__ = [
    "run_repl",
    "Source",
    "Sink",
]

Source = Callable[[], Awaitable[str | None]]

Sink = Callable[[str], None]


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

    The theatre owns the state and rendering bookkeeping so that callers need only raw
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

        assert turn_at(state) == HUMAN, f"expected human turn, got {turn_at(state)}"
        while True:
            text = await source()
            if text is None or text.strip() in ("exit", "quit"):
                return
            if not text.strip():
                continue
            state = append_only(state, message(HUMAN, text))
            state = append_only(state, turn_end(HUMAN, "relinquished"))
            break

        while turn_at(state) == ASSISTANT:
            async for item in to_events_and_live_updates(
                engine.run(Request(model=model, prompt=to_prompt(state), knobs=knobs))
            ):
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
                elif isinstance(item, (MessageUpdate, ThoughtUpdate)):
                    if not item.text:
                        continue
                    seen_ids.add(item.id)
                    if isinstance(item, ThoughtUpdate):
                        if not thinking_open:
                            sink("<thinking>")
                            thinking_open = True
                    elif thinking_open:
                        sink("</thinking>")
                        thinking_open = False
                    sink(item.text)
            state = append_only(state, turn_end(ASSISTANT, "relinquished"))
