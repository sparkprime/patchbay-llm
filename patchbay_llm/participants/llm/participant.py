"""The LLM participant (DESIGN2 §4).

One inference call per step: ``render`` the conversation state into a
``Prompt``, build a ``Request``, stream provider deltas through ``accumulate``
into journal ``Event``s, and yield them.  When the stream ends, yield
``turn_end`` to relinquish the floor -- the theatre's ``turn_at`` reads it back
and hands the floor to the other participant.

No tool dispatch yet: a tool batch is a future ``tools=`` config on this same
class, not a different one (DESIGN2 §4: the step is "one inference call, *or*
one batch of tool executions").  ``LlmParticipant(tools={})`` will be the
"no tools" config once a bash tool exists -- a config, not a class fork.
"""

from typing import AsyncIterator

from patchbay_llm.conversation import Conversation, turn_end
from patchbay_llm.events import Event
from patchbay_llm.infer_engine.engine import InferEngine
from patchbay_llm.infer_engine.request import Knobs, Request
from patchbay_llm.participants.llm.accumulate import accumulate
from patchbay_llm.participants.llm.render import render

__all__ = ["LlmParticipant"]


class LlmParticipant:
    """One inference call per step; constructor-injected with its engine and model."""

    def __init__(
        self,
        me: str,
        engine: InferEngine,
        model: str,
        knobs: Knobs = Knobs(),
    ) -> None:
        self.me = me
        self.engine = engine
        self.model = model
        self.knobs = knobs

    async def act(self, state: Conversation) -> AsyncIterator[Event]:
        """Render ``state`` to a prompt, run one inference call, yield its events."""
        prompt = render(state, self.me)
        request = Request(model=self.model, prompt=prompt, knobs=self.knobs)
        async for event in accumulate(self.engine.run(request), self.me):
            yield event
        yield turn_end(self.me, "relinquished")
