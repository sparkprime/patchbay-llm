"""The ACP theatre: a JSON-RPC consumer of the classic theatre shape.

Mirrors :mod:`repl_theatre`'s turn loop, replacing terminal I/O with ACP's
newline-delimited JSON-RPC wire protocol.  The conversation model is identical;
only the transport layer differs.

This module takes an abstract ``ReadLine``/``WriteLine`` pair rather than binding
to ``asyncio.StreamReader`` or OS stdio.  The caller (typically an example script)
wraps stdio in those callables.

No persistence, no cancellation, no tools, no usage reporting — parity with
:mod:`repl_theatre`'s own gaps.
"""

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from typing import (
    Any,
    Awaitable,
    Callable,
)

from patchbay_llm.classic_theatre.conversation import (
    ASSISTANT,
    HUMAN,
    Conversation,
    append_only,
    message,
    turn_at,
    turn_end,
)
from patchbay_llm.classic_theatre.seam import to_events_and_live_updates, to_prompt
from patchbay_llm.events import Event, Media
from patchbay_llm.infer_engine.engine import InferEngine
from patchbay_llm.infer_engine.request import Knobs, Request
from patchbay_llm.live import (
    LiveUpdate,
    MessageUpdate,
    ThoughtUpdate,
)

ReadLine = Callable[[], Awaitable[str | None]]
WriteLine = Callable[[str], Awaitable[None]]


@dataclass
class _Session:
    state: Conversation = ()
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


SessionId = str

sessions: dict[SessionId, _Session] = {}


def _notify(session_id: SessionId, update: dict[str, Any]) -> str:
    """Build an ACP notification line."""
    return json.dumps(
        {
            "method": "session/update",
            "params": {"sessionId": session_id, "update": update},
        }
    )


def _extract_prompt_text(blocks: list[dict[str, Any]]) -> str:
    """Extract text from ACP prompt blocks, ignoring non-text content."""
    parts: list[str] = []
    for b in blocks:
        bt = b.get("type")
        if bt == "text":
            parts.append(b.get("text", ""))
        elif bt == "resource_link":
            name = b.get("name", "")
            uri = b.get("uri", "")
            parts.append(f"[resource: {name} ({uri})]")
        else:
            parts.append(f"[unsupported content: {bt}]")
    return "".join(parts)


def _live_update_to_acp(item: LiveUpdate) -> dict[str, Any] | None:
    """Convert a LiveUpdate to an ACP update dict."""
    match item:
        case MessageUpdate(id=mid, text=t) if t:
            return {
                "sessionUpdate": "agent_message_chunk",
                "messageId": mid,
                "content": {"type": "text", "text": t},
            }
        case ThoughtUpdate(id=mid, text=t) if t:
            return {
                "sessionUpdate": "agent_thought_chunk",
                "messageId": mid,
                "content": {"type": "text", "text": t},
            }
        case _:
            return None


def _content_block(b: Media) -> dict[str, Any]:
    """Convert a Media block to an ACP ContentBlock."""
    return {"type": "text", "text": b.data.decode("utf-8")}


def _event_to_acp(item: Event) -> dict[str, Any] | None:
    """Convert an Event to an ACP update dict."""
    if item.meta.get("author") != ASSISTANT:
        return None
    if item.kind == "message":
        return {
            "sessionUpdate": "agent_message",
            "messageId": item.id,
            "content": [
                _content_block(b) for b in item.content if isinstance(b, Media)
            ],
        }
    if item.kind == "thought":
        return {
            "sessionUpdate": "agent_thought",
            "messageId": item.id,
            "content": [
                _content_block(b) for b in item.content if isinstance(b, Media)
            ],
        }
    return None


def _state_update_running() -> dict[str, Any]:
    return {"state": "running"}


def _state_update_idle(stop_reason: str) -> dict[str, Any]:
    return {"state": "idle", "stopReason": stop_reason}


async def _handle_prompt(
    session: _Session,
    session_id: SessionId,
    prompt_blocks: list[dict[str, Any]],
    engine: InferEngine,
    model: str,
    knobs: Knobs,
    write_line: WriteLine,
) -> dict[str, Any]:
    """Handle a session/prompt request — the turn loop."""
    async with session.lock:
        text = _extract_prompt_text(prompt_blocks)
        session.state = append_only(session.state, message(HUMAN, text))
        session.state = append_only(session.state, turn_end(HUMAN, "relinquished"))

        await write_line(_notify(session_id, _state_update_running()))

        stop_reason = "end_turn"
        while turn_at(session.state) == ASSISTANT:
            request = Request(model=model, prompt=to_prompt(session.state), knobs=knobs)
            async for item in to_events_and_live_updates(engine.run(request)):
                if isinstance(item, Event):
                    session.state = append_only(session.state, item)
                    update = _event_to_acp(item)
                else:
                    update = _live_update_to_acp(item)
                if update is not None:
                    await write_line(_notify(session_id, update))
            session.state = append_only(
                session.state, turn_end(ASSISTANT, "relinquished")
            )

        await write_line(_notify(session_id, _state_update_idle(stop_reason)))
        return {}


async def _handle_request(
    session_id: SessionId,
    request: dict[str, Any],
    engine: InferEngine,
    model: str,
    knobs: Knobs,
    write_line: WriteLine,
) -> dict[str, Any] | None:
    """Handle a JSON-RPC request. Returns a response dict or None."""
    method = request.get("method")
    params = request.get("params", {})
    req_id = request.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2",
                "info": {"name": "patchbay-llm", "version": "0.1.0"},
                "capabilities": {"session": {}},
            },
        }

    if method == "session/new":
        if "cwd" not in params:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32602, "message": "cwd is required"},
            }
        session_id = str(uuid.uuid4())
        sessions[session_id] = _Session()
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"sessionId": session_id, "configOptions": []},
        }

    if method == "session/prompt":
        session_id = params.get("sessionId", "")
        if session_id not in sessions:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32001, "message": "Unknown session"},
            }
        result = await _handle_prompt(
            sessions[session_id],
            session_id,
            params.get("prompt", []),
            engine,
            model,
            knobs,
            write_line,
        )
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": result,
        }

    if method == "session/cancel":
        # Accepted but ignored — no interrupt path
        return None

    if method == "session/close":
        session_id = params.get("sessionId", "")
        if session_id not in sessions:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32001, "message": "Unknown session"},
            }
        del sessions[session_id]
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {},
        }

    # Unknown method
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": "Method not found"},
    }


async def run_acp(
    read_line: ReadLine,
    write_line: WriteLine,
    engine: InferEngine,
    model: str,
    knobs: Knobs = Knobs(),
) -> None:
    """Run the ACP protocol.

    ``read_line`` returns one line of input (or None at EOF).
    ``write_line`` sends one line of output.
    """
    tasks: set[asyncio.Task[None]] = set()

    async def handler(line: str):
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            return  # Ignore malformed JSON

        response = await _handle_request(
            "",
            request,
            engine,
            model,
            knobs,
            write_line,
        )
        if response is not None and response.get("id") is not None:
            await write_line(json.dumps(response))

    while True:
        line = await read_line()
        if line is None:
            break
        task = asyncio.create_task(handler(line))
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    await asyncio.gather(*tasks)
