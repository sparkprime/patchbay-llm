"""
Demo: how to use the patchbay-llm inference API.

Run::

    export OPENROUTER_API_KEY=sk-or-v1-...
    ~/modular_agents/patchbay-llm/.venv/bin/python examples/infer_engine/demo.py

Covers the same ground as litellm_examples/ (basic streaming, usage/cost,
tools, structured output, reasoning) but through the single InferEngine.run
boundary -- no litellm or OpenAI field name appears in the call site.
"""

import asyncio
import json
import os
import sys
from typing import Any

from patchbay_llm.infer_engine.assemble import CacheHint, Contribution, assemble
from patchbay_llm.infer_engine.catalogue import Catalogue
from patchbay_llm.infer_engine.delta import Usage
from patchbay_llm.infer_engine.engine import InferEngine
from patchbay_llm.infer_engine.litellm import LitellmInferEngine
from patchbay_llm.infer_engine.prompt import Media, Message, Prompt, Result
from patchbay_llm.infer_engine.reply import collect
from patchbay_llm.infer_engine.request import Effort, Knobs, Request, Schema, Tool
from patchbay_llm.infer_engine.validation import check

DEFAULT_MODELS: dict[str, str] = {
    "claude": "openrouter/anthropic/claude-haiku-4.5",
    "gemini": "openrouter/google/gemini-2.5-flash-lite",
    "gpt": "openrouter/openai/gpt-4.1-mini",
}


def hr(title: str = "") -> None:
    """Print a section header."""
    line = "-" * 70
    print(f"\n{line}\n{title}\n{line}" if title else line)


def show_usage(usage: Usage, label: str = "") -> None:
    """Print a one-line summary of a Usage record."""
    prefix = f"[{label}] " if label else ""
    billed = f"{usage.billed}" if usage.billed is not None else "<none>"
    print(
        f"{prefix}model={usage.model} input={usage.input} cache_read={usage.cache_read} "
        f"cache_write={usage.cache_write} output={usage.output} thought={usage.thought} "
        f"estimated={usage.estimated} billed={billed}"
    )


def simple_prompt(text: str) -> Prompt:
    """A one-message user prompt, the smallest useful Prompt."""
    return Prompt(messages=(Message("user", (Media("text/plain", text.encode()),)),))


def build_engine() -> LitellmInferEngine:
    """Construct the engine with the default model aliases."""
    return LitellmInferEngine(models=DEFAULT_MODELS)


async def basic_streaming(engine: InferEngine, cat: Catalogue) -> None:
    """Same prompt, three providers, one call shape -- streamed."""
    hr("Basic streaming + usage/cost (one call shape, three providers)")
    for alias in DEFAULT_MODELS:
        facts = cat.facts(alias)
        print(f"{alias:8s} window={facts.window} max_output={facts.max_output}")
        req = Request(
            model=alias,
            prompt=simple_prompt("Name one moon of Jupiter. One word."),
            knobs=Knobs(max_output=16, temperature=0.0),
        )
        reply = await collect(engine.run(req))
        print(f"{alias:8s} -> {reply.text!r} stop={reply.stop.name}")
        show_usage(reply.usage, alias)


async def raw_delta_loop(engine: InferEngine) -> None:
    """Unrolled: see TextDelta / Finish arrive directly from the stream."""
    hr("Raw delta loop -- see TextDelta / Finish arrive directly")
    req = Request(
        model="claude",
        prompt=simple_prompt("Count from 1 to 5, one number per line."),
        knobs=Knobs(max_output=64),
    )
    n = 0
    async for delta in engine.run(req):
        n += 1
        print(f"  delta {n}: {delta}")
    print(f"({n} deltas total)")


async def tool_round_trip(engine: InferEngine) -> None:
    """Full round trip: request -> Call -> Result -> final answer."""
    hr("Full tool round trip: request -> Call -> Result -> final answer")
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
    reply = await collect(engine.run(req))
    print(f"stop={reply.stop.name} calls={[(c.tool, c.args) for c in reply.calls]}")

    def fake_weather(city: str, unit: str = "c") -> dict[str, Any]:
        return {
            "city": city,
            "temperature": 18 if unit == "c" else 64,
            "unit": unit,
            "condition": "cloudy",
        }

    results: list[Message] = []
    for call in reply.calls:
        res = fake_weather(**dict(call.args))
        results.append(
            Message(
                "user",
                (
                    Result(
                        call=call.id,
                        content=(Media("text/plain", json.dumps(res).encode()),),
                    ),
                ),
            )
        )
    # The tool loop is prompt.extend(reply.as_message(), *results) -- identity, no translation.
    prompt2 = prompt.extend(reply.as_message(), *results)
    req2 = Request(
        model="claude", prompt=prompt2, tools=(weather,), knobs=Knobs(max_output=200)
    )
    final = await collect(engine.run(req2))
    print(f"final answer: {final.text!r}")
    show_usage(final.usage, "claude/final")


async def structured_output(engine: InferEngine) -> None:
    """Structured output via Schema -- the reply IS a value."""
    hr("Structured output via Schema (the reply IS a value)")
    schema = Schema(
        name="recipe",
        schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "minutes": {"type": "integer"},
                "difficulty": {"type": "string", "enum": ["easy", "hard"]},
            },
            "required": ["name", "minutes", "difficulty"],
        },
    )
    req = Request(
        model="gpt",
        prompt=simple_prompt("Give a simple pasta recipe: difficulty and 2 steps."),
        output=schema,
        knobs=Knobs(max_output=400),
    )
    reply = await collect(engine.run(req))
    print(f"stop={reply.stop.name} text={reply.text!r}")


async def reasoning(engine: InferEngine) -> None:
    """Reasoning via Knobs.think -- the Effort enum normalises onto each provider."""
    hr("Reasoning via Knobs.think (Effort enum normalises onto each provider)")
    aliases = ("claude", "gpt")
    for alias in aliases:
        req = Request(
            model=alias,
            prompt=simple_prompt(
                "What is 17 * 23? Think it through, then give the final number."
            ),
            knobs=Knobs(max_output=1024, think=Effort.LOW),
        )
        reply = await collect(engine.run(req))
        print(f"{alias:8s} stop={reply.stop.name} thought_tokens={reply.usage.thought}")
        if alias == aliases[-1]:
            show_usage(reply.usage, "last")


async def assembly_and_check() -> None:
    """Assembly: Contributions -> Prompt with breakpoints; check() validates."""
    hr("Assembly: Contributions -> Prompt with breakpoints; check() validates")
    cs = [
        Contribution(
            directives=(Media("text/plain", b"You are concise."),),
            cache_hint=CacheHint.STABLE,
        ),
        Contribution(messages=(Message("user", (Media("text/plain", b"Hello."),)),)),
    ]
    prompt = assemble(cs, breakpoints=1)
    print("directives:", prompt.directives)
    print("messages:", prompt.messages)
    print("check:", check(prompt))


async def main() -> None:
    """Run every demo section in turn."""
    if not os.environ.get("OPENROUTER_API_KEY"):
        sys.exit("OPENROUTER_API_KEY is not set.")
    engine = build_engine()
    cat = engine.catalogue
    await basic_streaming(engine, cat)
    await raw_delta_loop(engine)
    await tool_round_trip(engine)
    await structured_output(engine)
    await reasoning(engine)
    await assembly_and_check()


if __name__ == "__main__":
    asyncio.run(main())
