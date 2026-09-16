"""``render``: the fold from conversation state to an inference ``Prompt``.

This is the "dumb" renderer (DESIGN2 §3 "Two vocabularies, one hub"): no
Sections, no compaction, no budget enforcement.  It groups complete content
events by consecutive author, collapses each author to an inference role
(``llm`` if it is ``me``, ``user`` otherwise -- DESIGN2 "Role collapse happens
in ``render``"), and converts each content block to an inference part via
``participants.llm.convert.to_part``.

``usage``, ``turn_start`` and ``turn_end`` are journal vocabulary with no
inference equivalent and are dropped.  ``budget`` is accepted but unused --
its signature slot is what lets a future Section-based assembler replace this
function without touching its callers (DESIGN2: "``render(state, budget) ->
Messages``").
"""

from typing import Sequence

from patchbay_llm.classic_theatre.participants.llm.convert import to_part
from patchbay_llm.events import Event
from patchbay_llm.infer_engine.prompt import Message, Prompt, Role

__all__ = ["render"]

_CONTENT_KINDS = ("message", "thought", "tool_call", "tool_result")


def _role(author: str, name: str) -> Role:
    return "llm" if author == name else "user"


def render(state: Sequence[Event], name: str, budget: int | None = None) -> Prompt:
    """Fold ``state`` into a ``Prompt`` for the participant identified by ``me``."""
    del budget  # unused -- signature slot for a future Section-based assembler.
    messages: list[Message] = []
    for e in state:
        if not e.complete or e.kind not in _CONTENT_KINDS:
            continue
        author = e.meta.get("author", "")
        role = _role(author, name)
        parts = tuple(to_part(b) for b in e.content)
        if messages and messages[-1].role == role:
            prev = messages[-1]
            messages[-1] = Message(role, prev.parts + parts)
        else:
            messages.append(Message(role, parts))
    return Prompt(messages=tuple(messages))
