"""ACP demo: wraps stdio, calls run_acp."""

import asyncio
import sys

from patchbay_llm.classic_theatre.acp_theatre import run_acp
from patchbay_llm.infer_engine.litellm import LitellmInferEngine
from patchbay_llm.infer_engine.request import Effort, Knobs

DEFAULT_MODELS: dict[str, str] = {
    "claude": "openrouter/anthropic/claude-haiku-4.5",
}


async def main() -> None:
    """Wire the engine and stdio transport; run the ACP loop."""
    engine = LitellmInferEngine(models=DEFAULT_MODELS)
    loop = asyncio.get_running_loop()

    async def read_line() -> str | None:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        return line.rstrip("\n") if line else None

    async def write_line(line: str) -> None:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()

    await run_acp(read_line, write_line, engine, "claude", Knobs(think=Effort.MEDIUM))


if __name__ == "__main__":
    asyncio.run(main())
