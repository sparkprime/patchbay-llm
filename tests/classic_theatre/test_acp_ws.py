"""Integration test: ACP frames survive the WebSocket round trip.

Exercises ``examples/classic_theatre/acp/ws_demo.py``'s handler against a real
``websockets`` client, asserting the same responses the stdio tests in
``test_acp_theatre.py`` expect — the point being that the WS wrapper transports
JSON-RPC frames unchanged. It does not duplicate the theatre's per-update
assertions; it only confirms frames round-trip cleanly.
"""

import importlib.util
import json
from pathlib import Path
from typing import Any, AsyncIterator

import pytest
import websockets
from websockets.datastructures import Headers
from websockets.http11 import Request, Response

from patchbay_llm.infer_engine.delta import Delta
from patchbay_llm.infer_engine.engine import InferEngine
from patchbay_llm.infer_engine.request import Request as EngineRequest


class FakeEngine(InferEngine):
    """Fake inference engine that yields a pre-built delta sequence."""

    def __init__(self, deltas: list[Delta]) -> None:
        self._deltas = deltas

    def run(self, request: EngineRequest) -> AsyncIterator[Delta]:
        """Yield the pre-built delta sequence, ignoring the request."""
        del request
        return self._iter()

    async def _iter(self) -> AsyncIterator[Delta]:
        for delta in self._deltas:
            yield delta


def _load_ws_demo():
    """Load the example wrapper module by path (examples/ isn't a package)."""
    path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "classic_theatre"
        / "acp"
        / "ws_demo.py"
    )
    spec = importlib.util.spec_from_file_location("ws_demo", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_ws_roundtrip_initialize_session_new_unknown() -> None:
    """initialize -> session/new -> unknown method round-trip over WS unchanged."""
    ws_demo = _load_ws_demo()
    engine = FakeEngine([])
    server = await websockets.serve(
        lambda ws: ws_demo._handler(ws, engine),  # pylint: disable=protected-access
        "127.0.0.1",
        0,
        subprotocols=[ws_demo.ACP_SUBPROTOCOL],
        select_subprotocol=ws_demo._select_subprotocol,  # pylint: disable=protected-access
        process_response=ws_demo._process_response,  # pylint: disable=protected-access
    )
    port = server.sockets[0].getsockname()[1]
    try:
        async with websockets.connect(
            f"ws://127.0.0.1:{port}",
            subprotocols=[ws_demo.ACP_SUBPROTOCOL],
        ) as ws:
            await ws.send(
                json.dumps(
                    {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
                )
            )
            await ws.send(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "session/new",
                        "params": {"cwd": "/tmp"},
                    }
                )
            )
            await ws.send(
                json.dumps(
                    {"jsonrpc": "2.0", "id": 3, "method": "session/list", "params": {}}
                )
            )

            responses: dict[int, dict[str, Any]] = {}
            for _ in range(3):
                msg = json.loads(await ws.recv())
                responses[msg["id"]] = msg

            init = responses[1]
            assert init["jsonrpc"] == "2.0"
            assert init["result"]["protocolVersion"] == "2"
            assert init["result"]["info"]["name"] == "patchbay-llm"
            assert "session" in init["result"]["capabilities"]

            sn = responses[2]
            assert sn["result"]["sessionId"] is not None
            assert sn["result"]["configOptions"] == []

            err = responses[3]
            assert err["error"]["code"] == -32601
    finally:
        server.close()
        await server.wait_closed()


def test_process_response_adds_pna_header() -> None:
    """The 101 upgrade response carries the Private Network Access allowance.

    Chrome checks the WS handshake response (not a separate OPTIONS preflight)
    for ``Access-Control-Allow-Private-Network: true``. Without it, a
    public-origin page is blocked from opening ws://localhost.
    """
    ws_demo = _load_ws_demo()
    request = Request(path="/", headers=Headers({"Origin": "https://acp-ui.github.io"}))
    response = Response(
        101,
        "Switching Protocols",
        Headers({"Upgrade": "websocket", "Connection": "Upgrade"}),
    )
    result = ws_demo._process_response(  # pylint: disable=protected-access
        websockets.ServerConnection.__new__(websockets.ServerConnection),
        request,
        response,
    )
    assert result is not None
    assert result.headers["Access-Control-Allow-Private-Network"] == "true"


@pytest.mark.asyncio
async def test_subprotocol_negotiated() -> None:
    """The server accepts the acp.v1 subprotocol the ACP client requests."""
    ws_demo = _load_ws_demo()
    engine = FakeEngine([])
    server = await websockets.serve(
        lambda ws: ws_demo._handler(ws, engine),  # pylint: disable=protected-access
        "127.0.0.1",
        0,
        subprotocols=[ws_demo.ACP_SUBPROTOCOL],
        select_subprotocol=ws_demo._select_subprotocol,  # pylint: disable=protected-access
        process_response=ws_demo._process_response,  # pylint: disable=protected-access
    )
    port = server.sockets[0].getsockname()[1]
    try:
        async with websockets.connect(
            f"ws://127.0.0.1:{port}",
            subprotocols=[ws_demo.ACP_SUBPROTOCOL],
        ) as ws:
            assert ws.subprotocol == ws_demo.ACP_SUBPROTOCOL
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_subprotocol_mismatch_logs_reason(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A client offering the wrong subprotocol is rejected with a logged reason.

    Without the custom ``select_subprotocol`` the library rejects at DEBUG
    (invisible under default logging); the wrapper surfaces a WARNING instead.
    """
    ws_demo = _load_ws_demo()
    engine = FakeEngine([])
    server = await websockets.serve(
        lambda ws: ws_demo._handler(ws, engine),  # pylint: disable=protected-access
        "127.0.0.1",
        0,
        subprotocols=[ws_demo.ACP_SUBPROTOCOL],
        select_subprotocol=ws_demo._select_subprotocol,  # pylint: disable=protected-access
        process_response=ws_demo._process_response,  # pylint: disable=protected-access
    )
    port = server.sockets[0].getsockname()[1]
    try:
        with caplog.at_level("WARNING", logger="acp_ws"):
            with pytest.raises(websockets.exceptions.InvalidStatus):
                async with websockets.connect(
                    f"ws://127.0.0.1:{port}",
                    subprotocols=["wrong.v9"],
                ):
                    pass
        assert any(
            "subprotocol negotiation failed" in r.message and "wrong.v9" in r.message
            for r in caplog.records
        )
    finally:
        server.close()
        await server.wait_closed()
