"""The boring seam between journal types and inference types.

Two families of translation live here:

- :func:`to_part` / :func:`from_part` -- complete content blocks ↔ inference
  parts.  ``events.py`` defines the journal's own ``Thought``/``ToolCall``/
  ``ToolResult`` and ``infer_engine.prompt`` defines the inference ``Part``
  union.  These are deliberately not unified: ``ToolCall`` keeps ``args`` as
  raw JSON text (possibly a fragment) on the journal side, while ``Call`` has
  parsed ``args`` on the inference side.  ``Media`` passes through unchanged
  -- it is the one shared type (defined in ``events.py``, imported by
  ``prompt.py``).

- :func:`to_live_update` -- inference deltas → live-update chunks.  The
  streaming counterpart: ``infer_engine.delta.Delta`` is the provider stream
  vocabulary, ``patchbay_llm.live.LiveUpdate`` is the neutral UI-facing
  vocabulary, and this is the only place that knows how to turn one into the
  other.
"""

import json
from typing import Any, Mapping

from patchbay_llm.events import (
    ContentBlock,
    EventId,
    Media,
    Thought,
    ToolCall,
    ToolResult,
)
from patchbay_llm.infer_engine.delta import CallDelta, Delta, TextDelta, ThoughtDelta
from patchbay_llm.infer_engine.prompt import Call, Part, Result
from patchbay_llm.infer_engine.prompt import Thought as InferThought
from patchbay_llm.live import LiveUpdate, MessageChunk, ThoughtChunk, ToolCallChunk

__all__ = ["to_part", "from_part", "to_live_update"]


def to_part(block: ContentBlock) -> Part:
    """Journal content block -> inference part.

    Only call this on a block from a *complete* Event: a partial ToolCall's
    ``args`` may not be valid JSON yet, and parsing it is the one place where
    "partial means unsafe" would bite.
    """
    match block:
        case Media():
            return block
        case Thought():
            return InferThought(block.text, block.signature)
        case ToolCall():
            parsed: Mapping[str, Any] = json.loads(block.args) if block.args else {}
            return Call(block.id, block.tool, parsed)
        case ToolResult():
            return Result(block.call, block.content, block.failed)
        case _:
            raise TypeError(f"unknown content block: {type(block).__name__}")


def from_part(part: Part) -> ContentBlock:
    """Inference part -> journal content block.

    The inverse direction.  ``Breakpoint`` has no journal equivalent -- it is
    a prompt-assembly concept placed by ``assemble``, not recorded content --
    and is rejected.
    """
    match part:
        case Media():
            return part
        case InferThought():
            return Thought(part.text, part.signature)
        case Call():
            return ToolCall(part.id, part.tool, json.dumps(part.args))
        case Result():
            return ToolResult(part.call, part.content, part.failed)
        case _:
            raise TypeError(f"unconvertible part: {type(part).__name__}")


def to_live_update(slot_id: EventId, delta: Delta) -> LiveUpdate:
    """Inference delta -> live-update chunk.

    The streaming counterpart to :func:`to_part`.  ``Delta`` is the
    per-chunk increment from the provider stream; :class:`LiveUpdate` is the
    neutral, UI-facing vocabulary.  ``slot_id`` is the :class:`EventId` the
    caller has assigned for this slot, stamped onto every chunk so consumers
    can correlate a live stream with the eventual :class:`Event`.
    """
    if isinstance(delta, TextDelta):
        return MessageChunk(id=slot_id, text=delta.text)
    if isinstance(delta, ThoughtDelta):
        return ThoughtChunk(id=slot_id, text=delta.text, signature=delta.signature)
    if isinstance(delta, CallDelta):
        return ToolCallChunk(
            id=slot_id, call_id=delta.id, tool=delta.tool, args=delta.args
        )
    raise TypeError(f"cannot convert {type(delta).__name__} to a LiveUpdate")
