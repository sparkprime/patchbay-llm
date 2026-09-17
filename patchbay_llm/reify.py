"""The LiveUpdate → ContentBlock seam.

:func:`reify_block` joins an ordered sequence of :class:`~patchbay_llm.live.LiveUpdate`
increments for one slot into the single :class:`~patchbay_llm.events.ContentBlock`
that the eventual :class:`~patchbay_llm.events.Event` will hold.  It is the
inverse direction of :func:`~patchbay_llm.infer_engine.convert.to_live_update`
(Delta → LiveUpdate) and the streaming counterpart of
:func:`~patchbay_llm.infer_engine.convert.from_part` (Part → ContentBlock):

    Delta   --convert.to_live_update-->  LiveUpdate  --reify_block-->  ContentBlock
    Part    --convert.from_part------->  ContentBlock

This module sits at the top level of ``patchbay_llm`` because it depends on
exactly the two top-level vocabularies it joins -- ``patchbay_llm.live`` (the
UI-facing streaming vocabulary) and ``patchbay_llm.events`` (the journal's
ground-truth vocabulary) -- and on nothing else.  It imports no
``infer_engine`` type and no theatre type, so it is callable from both live
accumulation (``accumulate``) and any recovery routine running in a different
process that needs to rejoin a persisted sequence of ``LiveUpdate`` records
into a ``ContentBlock``.

``patchbay_llm.live`` deliberately imports nothing from ``events`` (see its
docstring); the join therefore cannot live there.  ``events`` is the
substrate and should not depend on the live vocabulary either.  A dedicated
seam module at the top level is the one place that legitimately knows both.
"""

from typing import Sequence

from patchbay_llm.events import ContentBlock, Media, Thought, ToolCall, ToolResult
from patchbay_llm.live import (
    LiveUpdate,
    MessageChunk,
    ThoughtChunk,
    ToolCallChunk,
    ToolResultChunk,
)

__all__ = ["reify_block"]


def reify_block(kind: str, updates: Sequence[LiveUpdate]) -> ContentBlock:
    """Join ordered :class:`LiveUpdate` increments for one slot into a content block.

    Pure function of ``(kind, ordered updates) -> ContentBlock``, callable from
    both live accumulation and a recovery routine running in a different
    process.
    """
    if kind == "message":
        text = "".join(u.text for u in updates if isinstance(u, MessageChunk))
        return Media("text/plain", text.encode())

    if kind == "thought":
        text = "".join(u.text for u in updates if isinstance(u, ThoughtChunk))
        signature: str | None = None
        for u in updates:
            if isinstance(u, ThoughtChunk) and u.signature is not None:
                signature = u.signature
        return Thought(text, signature)

    if kind == "tool_call":
        call_id = ""
        tool = ""
        args = ""
        for u in updates:
            if isinstance(u, ToolCallChunk):
                if u.call_id is not None:
                    call_id = u.call_id
                if u.tool is not None:
                    tool = u.tool
                if u.args:
                    args += u.args
        return ToolCall(call_id, tool, args)

    if kind == "tool_result":
        call_id = ""
        text = ""
        failed = False
        for u in updates:
            if isinstance(u, ToolResultChunk):
                if u.call_id:
                    call_id = u.call_id
                if u.text:
                    text += u.text
                if u.failed is not None:
                    failed = u.failed
        return ToolResult(call_id, (Media("text/plain", text.encode()),), failed)

    raise ValueError(f"unknown slot kind: {kind!r}")
