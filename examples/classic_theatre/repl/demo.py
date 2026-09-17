"""
Demo: a minimal REPL theatre -- a back-and-forth conversation with an LLM
over a terminal.

Run::

    export OPENROUTER_API_KEY=sk-or-v1-...
    ~/modular_agents/patchbay-llm/.venv/bin/python examples/repl/demo.py

Type ``exit`` or ``quit`` (or Ctrl-D) to end the conversation cleanly.
"""

import asyncio
import os
import sys

from patchbay_llm.classic_theatre.repl_theatre import (
    run_repl,
    terminal_sink,
    terminal_source,
)
from patchbay_llm.infer_engine.litellm import LitellmInferEngine
from patchbay_llm.infer_engine.request import Effort, Knobs

DEFAULT_MODELS: dict[str, str] = {
    "claude": "openrouter/anthropic/claude-haiku-4.5",
}


async def main() -> None:
    """Wire the engine and theatre; run until EOF or exit."""
    if not os.environ.get("OPENROUTER_API_KEY"):
        sys.exit("OPENROUTER_API_KEY is not set.")
    engine = LitellmInferEngine(models=DEFAULT_MODELS)
    await run_repl(
        terminal_source(),
        terminal_sink,
        engine,
        "claude",
        Knobs(think=Effort.MEDIUM, max_output=8192),
    )


if __name__ == "__main__":
    asyncio.run(main())
