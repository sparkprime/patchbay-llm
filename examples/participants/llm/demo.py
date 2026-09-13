"""
Demo: the seam between infer_engine and the journal.

Run::

    export OPENROUTER_API_KEY=sk-or-v1-...
    ~/modular_agents/patchbay-llm/.venv/bin/python examples/participants/llm/demo.py

Two things this proves against a live provider:

1. ``accumulate`` turns a real streaming response into journal ``Event``s --
   growing partials, then complete events, then a usage event -- and the
   shapes match what the provider actually sends (slot assignment, fragment
   boundaries, etc.).  Unit tests cover this with synthetic ``Delta`` sequences;
   this is the first time the code sees a real stream.

2. The journal round-trip is equivalent to the direct path.  The same recorded
   delta sequence is fed through both::

       collect() -> Reply.as_message()           (the direct infer_engine path)
       accumulate() -> convert.to_part()         (the journal round-trip path)

   and the resulting ``Message`` objects must be identical.  This is the
   property that lets a future ``LlmParticipant`` build its next prompt from
   journalled events without losing anything the direct path would have kept.

Deltas are recorded once from a single live call and replayed to both paths,
so non-determinism is taken out of the equation -- the comparison is exact.
"""

import asyncio
import os
import sys
from typing import AsyncIterator

from patchbay_llm.events import Event, Media
from patchbay_llm.infer_engine.delta import Delta
from patchbay_llm.infer_engine.litellm import LitellmInferEngine
from patchbay_llm.infer_engine.prompt import Message, Part, Prompt
from patchbay_llm.infer_engine.reply import collect
from patchbay_llm.infer_engine.request import Knobs, Request, Tool
from patchbay_llm.participants.llm.accumulate import accumulate
from patchbay_llm.participants.llm.convert import to_part

DEFAULT_MODELS: dict[str, str] = {
    "claude": "openrouter/anthropic/claude-haiku-4.5",
}


def hr(title: str = "") -> None:
    """Print a section header."""
    line = "-" * 70
    print(f"\n{line}\n{title}\n{line}" if title else line)


def simple_prompt(text: str) -> Prompt:
    """A one-message user prompt, the smallest useful Prompt."""
    return Prompt(messages=(Message("user", (Media("text/plain", text.encode()),)),))


async def _replay(deltas: list[Delta]) -> AsyncIterator[Delta]:
    """Replay a recorded list of deltas as an async iterator."""
    for d in deltas:
        yield d


async def _record_deltas(engine: LitellmInferEngine, req: Request) -> list[Delta]:
    """Run a request to completion, recording every delta."""
    deltas: list[Delta] = []
    async for d in engine.run(req):
        deltas.append(d)
    return deltas


def _content_preview(event: Event) -> str:
    """A short, readable preview of one event's content."""
    if not event.content:
        meta = ", ".join(f"{k}={v}" for k, v in event.meta.items() if k != "author")
        return f"(meta: {meta})"
    block = event.content[0]
    if isinstance(block, Media) and block.mime == "text/plain":
        return repr(block.data.decode("utf-8", errors="replace"))
    return repr(block)[:60]


async def accumulate_loop(engine: LitellmInferEngine) -> None:
    """See journal Events form live from a real provider stream."""
    hr("accumulate: live provider stream -> journal Events")
    req = Request(
        model="claude",
        prompt=simple_prompt("Count from 1 to 5, one number per line."),
        knobs=Knobs(max_output=64),
    )
    n = 0
    async for event in accumulate(engine.run(req), "llm:main"):
        n += 1
        flag = "C" if event.complete else "."
        preview = _content_preview(event)
        print(
            f"  event {n:3d} [{flag}] kind={event.kind:10s} id={event.id[:8]} {preview}"
        )
    print(f"({n} events total)")


def _events_to_llm_message(events: list[Event]) -> Message:
    """Build the llm Message from accumulated events (the journal path).

    Skips partials (``complete=False``) and the terminal ``usage`` event,
    keeping only the final, complete content events -- one per slot.
    """
    parts: list[Part] = []
    for e in events:
        if e.kind == "usage" or not e.complete:
            continue
        for block in e.content:
            parts.append(to_part(block))
    return Message("llm", tuple(parts))


async def journal_round_trip_equivalence(engine: LitellmInferEngine) -> None:
    """Prove accumulate+convert produces the same Message as collect+Reply."""
    hr("Equivalence: collect/Reply.as_message == accumulate/convert.to_part")
    weather = Tool(
        name="get_weather",
        description="Get the current weather for a city.",
        params={
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string", "enum": ["c", "f"]},
            },
            "required": ["city"],
        },
    )
    prompt = simple_prompt("What's the weather in Paris? Use the tool.")
    req = Request(
        model="claude", prompt=prompt, tools=(weather,), knobs=Knobs(max_output=300)
    )

    # One live call, recorded deltas shared by both paths. Recording once
    # takes LLM non-determinism out of the equation -- the comparison is exact.
    deltas = await _record_deltas(engine, req)
    print(f"  recorded {len(deltas)} deltas from one live call")

    # Path A: the direct infer_engine path.
    reply = await collect(_replay(deltas))
    direct_msg = reply.as_message()
    print(
        f"  collect()       -> {len(reply.parts)} parts, "
        f"{len(reply.calls)} call(s): {[(c.tool, dict(c.args)) for c in reply.calls]}"
    )

    # Path B: the journal round-trip path.
    events = [e async for e in accumulate(_replay(deltas), "llm:main")]
    journal_msg = _events_to_llm_message(events)
    complete = [e for e in events if e.complete and e.kind != "usage"]
    print(
        f"  accumulate()    -> {len(complete)} complete content events, "
        f"{len(journal_msg.parts)} parts"
    )

    # The assertion that matters: both paths produce the same Prompt message.
    if direct_msg == journal_msg:
        print("  PASS: both paths produce identical Messages")
    else:
        print("  FAIL: paths diverge!")
        print(f"    direct:  {direct_msg}")
        print(f"    journal: {journal_msg}")
        raise SystemExit(1)


async def main() -> None:
    """Run both demo sections in turn."""
    if not os.environ.get("OPENROUTER_API_KEY"):
        sys.exit("OPENROUTER_API_KEY is not set.")
    engine = LitellmInferEngine(models=DEFAULT_MODELS)
    await accumulate_loop(engine)
    await journal_round_trip_equivalence(engine)


if __name__ == "__main__":
    asyncio.run(main())
