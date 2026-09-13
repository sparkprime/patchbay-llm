"""The boring seam between journal content and inference parts.

Lives here (inside ``infer_engine/``) rather than in ``events.py`` because the
subdirectory can import from the directory above it, and the journal cannot.
``events.py`` defines the journal's own ``Thought``/``ToolCall``/``ToolResult``
and this module is the only place that knows how to translate between those
and the inference ``Part`` union.  ``Media`` passes through unchanged --
it is the one shared type.
"""

import json
from typing import Any, Mapping

from patchbay_llm.events import ContentBlock, Media, Thought, ToolCall, ToolResult
from patchbay_llm.infer_engine.prompt import Call, Part, Result
from patchbay_llm.infer_engine.prompt import Thought as InferThought

__all__ = ["to_part", "from_part"]


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
