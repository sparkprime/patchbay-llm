"""The LLM participant (DESIGN2 §4).

One inference call per step: ``render`` the conversation state into a
``Prompt``, build a ``Request``, stream provider deltas through ``accumulate``
into ``Event | PartialEvent`` items, and yield them.  When the stream ends,
yield ``turn_end`` to relinquish the floor -- the theatre's ``turn_at`` reads
it back and hands the floor to the other participant.

The ``sink`` parameter is an optional ``DurabilitySink`` (durable_partials.md:
"Durability"), constructor-injected per DESIGN2 principle 2.  ``None``/no-op
for tests and for theatres that accept losing an in-flight turn.

No tool dispatch yet: a tool batch is a future ``tools=`` config on this same
class, not a different one (DESIGN2 §4: the step is "one inference call, *or*
one batch of tool executions").  ``LlmParticipant(tools={})`` will be the
"no tools" config once a bash tool exists -- a config, not a class fork.
"""

from typing import AsyncIterator, Union

from patchbay_llm.conversation import Conversation, turn_end
from patchbay_llm.events import Event
from patchbay_llm.infer_engine.engine import InferEngine
from patchbay_llm.infer_engine.request import Knobs, Request
from patchbay_llm.participants.llm.accumulate import (
    DurabilitySink,
    PartialEvent,
    accumulate,
)
from patchbay_llm.participants.llm.render import render

__all__ = ["LlmParticipant"]

Item = Union[Event, PartialEvent]


class LlmParticipant:
    """One inference call per step; constructor-injected with its engine and model."""

    def __init__(
        self,
        me: str,
        engine: InferEngine,
        model: str,
        knobs: Knobs = Knobs(),
        sink: DurabilitySink | None = None,
    ) -> None:
        self.me = me
        self.engine = engine
        self.model = model
        self.knobs = knobs
        self.sink = sink

    async def act(self, state: Conversation) -> AsyncIterator[Item]:
        """Render ``state`` to a prompt, run one inference call, yield its items.

        Yields ``Event | PartialEvent`` (durable_partials.md): a
        ``PartialEvent`` handle once per new slot, then final ``Event`` s on
        ``Finish``, then ``turn_end``.
        """
        prompt = render(state, self.me)
        request = Request(model=self.model, prompt=prompt, knobs=self.knobs)
        async for item in accumulate(self.engine.run(request), self.me, self.sink):
            yield item
        yield turn_end(self.me, "relinquished")
