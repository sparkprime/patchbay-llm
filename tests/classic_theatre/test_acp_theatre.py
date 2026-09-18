"""End-to-end tests for ACP theatre."""

import asyncio
import json
from typing import Any, AsyncIterator

import pytest

from patchbay_llm.classic_theatre.acp_theatre import run_acp
from patchbay_llm.infer_engine.delta import Delta, Finish, Stop, TextDelta, Usage
from patchbay_llm.infer_engine.engine import InferEngine
from patchbay_llm.infer_engine.request import Request


class FakeEngine(InferEngine):
    """Fake inference engine that yields a pre-built delta sequence."""

    def __init__(self, deltas: list[Delta]) -> None:
        self._deltas = deltas

    def run(self, request: Request) -> AsyncIterator[Delta]:
        """Yield the pre-built delta sequence, ignoring the request."""
        del request
        return self._iter()

    async def _iter(self) -> AsyncIterator[Delta]:
        for delta in self._deltas:
            yield delta


class _Capturer:
    """Captures write_line calls into a list."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    async def capture(self, line: str) -> None:
        """Append a line to the capture list."""
        self.lines.append(line)


def _strip_lines(lines: list[str]) -> list[dict[str, Any]]:
    """Parse JSON lines from the capture."""
    result: list[dict[str, Any]] = []
    for line in lines:
        if line.strip():
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return result


@pytest.mark.asyncio
async def test_run_acp_initialize() -> None:
    """Initialize request returns protocol version and capabilities."""
    cap = _Capturer()
    input_queue: asyncio.Queue[str | None] = asyncio.Queue()
    input_queue.put_nowait(
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    )
    input_queue.put_nowait(None)

    async def read_line() -> str | None:
        return await input_queue.get()

    await run_acp(
        read_line,
        cap.capture,
        FakeEngine([]),
        "claude",
    )

    lines = _strip_lines(cap.lines)
    assert len(lines) == 1
    assert lines[0]["jsonrpc"] == "2.0"
    assert lines[0]["id"] == 1
    result = lines[0]["result"]
    assert result["protocolVersion"] == "2"
    assert result["info"]["name"] == "patchbay-llm"
    assert "session" in result["capabilities"]


@pytest.mark.asyncio
async def test_run_acp_session_new() -> None:
    """session/new creates a session and returns sessionId."""
    cap = _Capturer()
    input_queue: asyncio.Queue[str | None] = asyncio.Queue()
    input_queue.put_nowait(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "session/new",
                "params": {"cwd": "/tmp"},
            }
        )
    )
    input_queue.put_nowait(None)

    async def read_line() -> str | None:
        return await input_queue.get()

    await run_acp(
        read_line,
        cap.capture,
        FakeEngine([]),
        "claude",
    )

    lines = _strip_lines(cap.lines)
    assert len(lines) == 1
    assert lines[0]["result"]["sessionId"] is not None
    assert lines[0]["result"]["configOptions"] == []


@pytest.mark.asyncio
async def test_run_acp_method_not_found() -> None:
    """Unknown methods return method not found error."""
    cap = _Capturer()
    input_queue: asyncio.Queue[str | None] = asyncio.Queue()
    input_queue.put_nowait(
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "session/list", "params": {}})
    )
    input_queue.put_nowait(None)

    async def read_line() -> str | None:
        return await input_queue.get()

    await run_acp(
        read_line,
        cap.capture,
        FakeEngine([]),
        "claude",
    )

    lines = _strip_lines(cap.lines)
    assert len(lines) == 1
    assert lines[0]["error"]["code"] == -32601


@pytest.mark.asyncio
async def test_run_acp_session_prompt_roundtrip() -> None:
    """session/new then session/prompt with the returned sessionId succeeds."""
    cap = _Capturer()
    input_queue: asyncio.Queue[str | None] = asyncio.Queue()

    deltas: list[Delta] = [
        TextDelta(slot="text", text="hello"),
        Finish(
            stop=Stop.END,
            usage=Usage(
                model="m", input=10, cache_read=0, cache_write=0, output=5, thought=0
            ),
        ),
    ]

    async def read_line() -> str | None:
        return await input_queue.get()

    async def send_prompt_after_new() -> None:
        while not cap.lines:
            await asyncio.sleep(0.01)
        new_response = json.loads(cap.lines[0])
        sid = new_response["result"]["sessionId"]
        input_queue.put_nowait(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "session/prompt",
                    "params": {
                        "sessionId": sid,
                        "prompt": [{"type": "text", "text": "hi"}],
                    },
                }
            )
        )
        input_queue.put_nowait(None)

    input_queue.put_nowait(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "session/new",
                "params": {"cwd": "/tmp"},
            }
        )
    )

    task = asyncio.create_task(send_prompt_after_new())

    await run_acp(
        read_line,
        cap.capture,
        FakeEngine(deltas),
        "claude",
    )
    await task

    lines = _strip_lines(cap.lines)
    # session/new response + notifications + session/prompt response
    assert len(lines) >= 2
    new_response = lines[0]
    assert new_response["id"] == 1
    sid = new_response["result"]["sessionId"]
    assert sid  # non-empty
    # Find the session/prompt response (has id: 2); notifications have method instead
    prompt_responses = [l for l in lines if l.get("id") == 2]
    assert len(prompt_responses) == 1
    assert "error" not in prompt_responses[0]


@pytest.mark.asyncio
async def test_run_acp_session_prompt_unknown_session() -> None:
    """session/prompt with an unknown sessionId returns -32001."""
    cap = _Capturer()
    input_queue: asyncio.Queue[str | None] = asyncio.Queue()
    input_queue.put_nowait(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "session/prompt",
                "params": {
                    "sessionId": "nope",
                    "prompt": [{"type": "text", "text": "hi"}],
                },
            }
        )
    )
    input_queue.put_nowait(None)

    async def read_line() -> str | None:
        return await input_queue.get()

    await run_acp(
        read_line,
        cap.capture,
        FakeEngine([]),
        "claude",
    )

    lines = _strip_lines(cap.lines)
    assert len(lines) == 1
    assert lines[0]["error"]["code"] == -32001
