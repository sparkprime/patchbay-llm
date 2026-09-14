"""The human participant (DESIGN2 §4).

One message per step: block on ``source`` (a terminal read or a future ACP
inbox queue), yield a ``message`` event, then ``turn_end`` to hand the floor
back.  The terminal case and a future ACP-queue case are two different
``source`` callables plugged into the same class, not two classes (DESIGN2's
own participant table describes one ``HumanParticipant`` that differs only in
what it blocks on -- "a terminal read or a protocol queue").

``source`` returns ``str | None``: ``None`` signals end-of-input (EOF on a
terminal), and the participant yields only a ``turn_end`` so the theatre can
wind down cleanly.
"""

import asyncio
from typing import AsyncIterator, Awaitable, Callable

from patchbay_llm.conversation import Conversation, message, turn_end
from patchbay_llm.events import Event

__all__ = ["HumanParticipant", "terminal_source", "Source"]

Source = Callable[[], Awaitable[str | None]]


class HumanParticipant:
    """One message per step; constructor-injected with a ``source`` callable."""

    def __init__(self, me: str, source: Source) -> None:
        self.me = me
        self.source = source

    async def act(self, state: Conversation) -> AsyncIterator[Event]:
        """Read one non-empty line from ``source``, yield ``message`` then ``turn_end``."""
        del state  # the human does not consult state to produce its message.
        while True:
            text = await self.source()
            if text is None:
                yield turn_end(self.me, "relinquished")
                return
            if text.strip():
                yield message(self.me, text)
                yield turn_end(self.me, "relinquished")
                return


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
