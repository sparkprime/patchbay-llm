"""Turn finished live updates into a single content block for an event."""

from typing import Sequence

from patchbay_llm.events import ContentBlock, Media, Thought, ToolCall, ToolResult
from patchbay_llm.live import (
    LiveUpdate,
    MessageUpdate,
    ThoughtUpdate,
    ToolCallUpdate,
    ToolResultUpdate,
)

__all__ = ["freeze_block"]


def freeze_block(kind: str, updates: Sequence[LiveUpdate]) -> ContentBlock:
    """Join ordered `LiveUpdate` increments for one slot into a content block."""
    if kind == "message":
        text = "".join(u.text for u in updates if isinstance(u, MessageUpdate))
        return Media("text/plain", text.encode())

    if kind == "thought":
        text = "".join(u.text for u in updates if isinstance(u, ThoughtUpdate))
        signature: str | None = None
        for u in updates:
            if isinstance(u, ThoughtUpdate) and u.signature is not None:
                signature = u.signature
        return Thought(text, signature)

    if kind == "tool_call":
        call_id = ""
        tool = ""
        args = ""
        for u in updates:
            if isinstance(u, ToolCallUpdate):
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
            if isinstance(u, ToolResultUpdate):
                if u.call_id:
                    call_id = u.call_id
                if u.text:
                    text += u.text
                if u.failed is not None:
                    failed = u.failed
        return ToolResult(call_id, (Media("text/plain", text.encode()),), failed)

    raise ValueError(f"unknown slot kind: {kind!r}")
