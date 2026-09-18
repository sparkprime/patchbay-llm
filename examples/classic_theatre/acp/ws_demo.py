"""ACP demo over WebSocket: a persistent server reusing ``run_acp`` unmodified.

Sibling to ``demo.py`` (the stdio variant). The heavy import
(``LitellmInferEngine`` / ``litellm``) happens once at startup; every accepted
WebSocket connection then reuses the warm engine as a cheap coroutine via
``run_acp``, instead of websocketd's per-connection process fork.

Run::

    export OPENROUTER_API_KEY=sk-or-v1-...
    ~/modular_agents/patchbay-llm/.venv/bin/python examples/classic_theatre/acp/ws_demo.py

Then point an ACP web UI at ``ws://localhost:8765``.
"""

import asyncio
import logging
import os
import sys
from collections.abc import Sequence

import websockets
from websockets.http11 import Request, Response

from patchbay_llm.classic_theatre.acp_theatre import run_acp
from patchbay_llm.infer_engine.litellm import LitellmInferEngine
from patchbay_llm.infer_engine.request import Effort, Knobs

DEFAULT_MODELS: dict[str, str] = {
    "claude": "openrouter/anthropic/claude-haiku-4.5",
}

ACP_SUBPROTOCOL = "acp.v1"

log = logging.getLogger("acp_ws")


def _select_subprotocol(
    _conn: websockets.ServerConnection, subprotocols: Sequence[str]
) -> str | None:
    """Negotiate the ACP subprotocol, logging a clear reason on mismatch.

    The library's default negotiation rejects a mismatched subprotocol with a
    400 but only logs the reason at DEBUG (gated on the logger's effective
    level), so with default logging the failure is silent. This surfaces a
    human-readable message at WARNING so a failed connection isn't invisible.
    """
    offered = set(subprotocols)
    if ACP_SUBPROTOCOL in offered:
        return ACP_SUBPROTOCOL
    if offered:
        log.warning(
            "subprotocol negotiation failed: client offered %s, server requires %s",
            sorted(offered),
            ACP_SUBPROTOCOL,
        )
    else:
        log.warning(
            "subprotocol negotiation failed: client offered no subprotocol, "
            "server requires %s",
            ACP_SUBPROTOCOL,
        )
    raise websockets.exceptions.NegotiationError(
        f"invalid subprotocol; expected {ACP_SUBPROTOCOL}"
    )


def _process_response(
    _conn: websockets.ServerConnection,
    _request: Request,
    response: Response,
) -> Response | None:
    """Add the Private Network Access allowance to the 101 response.

    Chrome does not preflight WebSocket with a separate OPTIONS; it checks the
    upgrade response itself for ``Access-Control-Allow-Private-Network``.
    Without it, a public-origin page (e.g. https://acp-ui.github.io) is blocked
    from opening ws://localhost — the connection fails with status 0 before any
    handler runs. This is transport-layer browser policy, not auth.
    """
    response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response


async def _handler(ws: websockets.ServerConnection, engine: LitellmInferEngine) -> None:
    """One WS connection -> one ``run_acp`` call -> its own local sessions dict."""

    async def read_line() -> str | None:
        try:
            data = await ws.recv()
        except websockets.ConnectionClosed:
            return None
        return data if isinstance(data, str) else data.decode()

    await run_acp(read_line, ws.send, engine, "claude", Knobs(think=Effort.MEDIUM))


async def main() -> None:
    """Build the engine once, then serve connections until killed."""
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    if not os.environ.get("OPENROUTER_API_KEY"):
        sys.exit("OPENROUTER_API_KEY is not set.")
    engine = LitellmInferEngine(models=DEFAULT_MODELS)
    host = os.environ.get("WS_HOST", "localhost")
    port = int(os.environ.get("WS_PORT", "8765"))
    async with websockets.serve(
        lambda ws: _handler(ws, engine),
        host,
        port,
        subprotocols=[ACP_SUBPROTOCOL],
        select_subprotocol=_select_subprotocol,
        process_response=_process_response,
    ):
        print(f"ACP over WebSocket on ws://{host}:{port}", file=sys.stderr)
        await asyncio.Future()  # run until killed


if __name__ == "__main__":
    asyncio.run(main())
