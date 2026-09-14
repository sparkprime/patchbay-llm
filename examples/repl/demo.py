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

from patchbay_llm.infer_engine.litellm import LitellmInferEngine
from patchbay_llm.infer_engine.request import Effort, Knobs
from patchbay_llm.participants.human import HumanParticipant, terminal_source
from patchbay_llm.participants.llm.participant import LlmParticipant
from patchbay_llm.repl_theatre import run_repl

DEFAULT_MODELS: dict[str, str] = {
    "claude": "openrouter/anthropic/claude-haiku-4.5",
}


async def main() -> None:
    """Wire the engine, participants and theatre; run until EOF or exit."""
    if not os.environ.get("OPENROUTER_API_KEY"):
        sys.exit("OPENROUTER_API_KEY is not set.")
    engine = LitellmInferEngine(models=DEFAULT_MODELS)
    human = HumanParticipant("human:you", terminal_source())
    llm = LlmParticipant(
        "llm:main", engine, "claude", Knobs(think=Effort.MEDIUM, max_output=8192)
    )
    await run_repl(human, llm)


if __name__ == "__main__":
    asyncio.run(main())
