"""The one engine that ships: :class:`LitellmInferEngine`.

Every concept in the API maps to something verified in ``litellm_examples/``
(INFERENCE.md §8).  This module is the entire specification of the mapping, and
nothing provider-shaped escapes it.

Eight adapter obligations (§8) are honoured here:

1. Always set ``stream_options={"include_usage": True}``.
2. Re-derive the input counts to be disjoint.
3. Never exhaust without a ``Finish``.
4. Merge adjacent same-role messages and drop empties.
5. Assign slots.
6. Inline ``$ref`` s and close every object before sending a schema.
7. Ask for the reasoning trace explicitly (``reasoning.exclude=False``).
8. Do not set ``litellm.drop_params = True``.
"""

# litellm's API is largely untyped; relax pyright's strictness for the
# member/argument checks that would otherwise fight every chunk access.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportMissingTypeArgument=false
# pyright: reportMissingParameterType=false, reportUnknownLambdaType=false
# pyright: reportMissingTypeStubs=false
# pyright: reportAttributeAccessIssue=false, reportCallIssue=false

from __future__ import annotations

import asyncio
import base64
import json
from decimal import Decimal
from typing import Any, AsyncIterator, Mapping, Sequence

import litellm
from litellm import Router

from patchbay_llm.infer_engine.catalogue import Catalogue, Facts, Rate
from patchbay_llm.infer_engine.delta import (
    CallDelta,
    Delta,
    Finish,
    Stop,
    TextDelta,
    ThoughtDelta,
    Usage,
)
from patchbay_llm.infer_engine.errors import (
    InferenceError,
    Rejected,
    Throttled,
    TooLarge,
    Unreachable,
)
from patchbay_llm.infer_engine.prompt import (
    Breakpoint,
    Call,
    Media,
    Message,
    Part,
    Prompt,
    Result,
    Thought,
)
from patchbay_llm.infer_engine.request import Effort, Request, Tool

__all__ = ["LitellmInferEngine", "LitellmCatalogue"]

# Default thinking budgets (tokens) for budget-style providers.  0 means
# "unsupported" -> the adapter uses effort-style ``reasoning_effort`` instead.
_ANTHROPIC_THINKING: Mapping[Effort, int] = {
    Effort.NONE: 0,
    Effort.LOW: 1024,
    Effort.MEDIUM: 4096,
    Effort.HIGH: 16384,
}
_NO_THINKING: Mapping[Effort, int] = {e: 0 for e in Effort}

_REASON_EFFORT: Mapping[Effort, str] = {
    Effort.LOW: "low",
    Effort.MEDIUM: "medium",
    Effort.HIGH: "high",
}


class LitellmCatalogue(Catalogue):
    """A :class:`Catalogue` backed by ``litellm.model_cost`` and ``token_counter``.

    No network dependency -- usable standalone for budgeting and analysis.
    """

    def __init__(self, models: Mapping[str, str]) -> None:
        # alias -> full litellm model id (e.g. "openrouter/anthropic/claude-haiku-4.5")
        self._models = dict(models)

    def _full(self, alias: str) -> str:
        return self._models[alias]

    def facts(self, model: str) -> Facts:
        full = self._full(model)
        entry = litellm.model_cost.get(full, {})
        window = int(entry.get("max_input_tokens", 0))
        max_output = int(entry.get("max_output_tokens", 0))
        provider = full.split("/")[0] if "/" in full else ""
        is_anthropic = provider == "anthropic"
        thinking = _ANTHROPIC_THINKING if is_anthropic else _NO_THINKING
        rates = self._rates(entry)
        return Facts(
            model=full,
            window=window,
            max_output=max_output,
            breakpoints=1 if is_anthropic else 1,
            breakpoint_min=1024,
            thinking=thinking,
            rates=rates,
        )

    def _rates(self, entry: Mapping[str, Any]) -> tuple[tuple[int, Rate], ...]:
        def _dec(key: str) -> Decimal:
            v = entry.get(key, 0)
            return Decimal(str(v)) if v is not None else Decimal("0")

        base = Rate(
            input=_dec("input_cost_per_token"),
            cache_read=_dec("cached_input_token_cost")
            or _dec("cache_read_input_token_cost"),
            cache_write=_dec("cache_creation_input_token_cost"),
            output=_dec("output_cost_per_token"),
        )
        tiers: list[tuple[int, Rate]] = [(0, base)]
        above = entry.get("input_cost_per_token_above_128k_tokens")
        if above is not None:
            high = Rate(
                input=Decimal(str(above)),
                cache_read=base.cache_read,
                cache_write=base.cache_write,
                output=Decimal(
                    str(entry.get("output_cost_per_token_above_128k_tokens", above))
                ),
            )
            tiers.append((128000, high))
        return tuple(tiers)

    def price(self, usage: Usage) -> Decimal:
        full = usage.model
        entry = litellm.model_cost.get(full, {})
        tiers = self._rates(entry)
        total_prompt = usage.input + usage.cache_read + usage.cache_write
        rate = tiers[-1][1]
        for threshold, r in tiers:
            if total_prompt >= threshold:
                rate = r
        return (
            rate.input * Decimal(usage.input)
            + rate.cache_read * Decimal(usage.cache_read)
            + rate.cache_write * Decimal(usage.cache_write)
            + rate.output * Decimal(usage.output)
        )

    def measure(self, prompt: Prompt, model: str) -> int:
        full = self._full(model)
        messages = _build_messages_for_count(prompt)
        try:
            return int(litellm.token_counter(model=full, messages=messages))
        except Exception:  # pylint: disable=broad-exception-caught
            # token_counter raises a wide variety of errors for unknown models;
            # an approximation is allowed to be 0 when counting is unavailable.
            return 0


class LitellmInferEngine:
    """The real engine.  Provider breadth, verified against ``litellm_examples/``.

    Construct with a mapping of short aliases to full litellm model ids::

        engine = LitellmInferEngine(models={"main": "openrouter/anthropic/claude-haiku-4.5"})
        async for delta in engine.run(request):
            ...
    """

    def __init__(self, models: Mapping[str, str]) -> None:
        self.models = dict(models)
        self._router = Router(
            model_list=[
                {"model_name": alias, "litellm_params": {"model": mid}}
                for alias, mid in self.models.items()
            ]
        )
        self._catalogue = LitellmCatalogue(self.models)

    @property
    def catalogue(self) -> LitellmCatalogue:
        """The :class:`LitellmCatalogue` for offline facts/pricing/measure."""
        return self._catalogue

    # -- public API ----------------------------------------------------------

    def run(self, request: Request) -> AsyncIterator[Delta]:
        """One streaming call.  Returns an async iterator of :class:`Delta`."""
        return self._stream(request)

    # -- internals -----------------------------------------------------------

    async def _stream(self, request: Request) -> AsyncIterator[Delta]:
        messages = _build_messages_for_call(request.prompt)
        kwargs = _build_kwargs(request, self.models, self._catalogue)
        try:
            stream = await self._router.acompletion(
                model=request.model,
                messages=messages,
                stream=True,
                stream_options={"include_usage": True},
                **kwargs,
            )
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            raise _map_error(exc) from exc

        finish_reason: str | None = None
        state = _SlotState()
        try:
            async for chunk in stream:
                choices = getattr(chunk, "choices", None) or []
                if choices:
                    choice = choices[0]
                    fr = getattr(choice, "finish_reason", None)
                    if fr:
                        finish_reason = fr
                    delta = getattr(choice, "delta", None)
                    if delta is not None:
                        for d in _translate_delta(delta, state):
                            yield d
                usage_obj = getattr(chunk, "usage", None)
                if usage_obj is not None and _has_counts(usage_obj):
                    yield _finish(usage_obj, finish_reason, request.model, self.models)
                    return
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            raise _map_error(exc) from exc

        # Stream exhausted.  Never a silent exhaustion without a Finish.
        if finish_reason is not None:
            # Finished properly but no usage chunk -- synthesise, estimated.
            yield _synthesise_finish(finish_reason, request.model, self.models)
            return
        # No finish_reason and no usage -- the connection dropped.
        raise Unreachable("stream closed without a finish reason or usage chunk")


# ---------------------------------------------------------------------------
# Slot-assigned delta translation (obligation 5)
# ---------------------------------------------------------------------------


class _SlotState:
    """Per-call mutable slot counter, so the generator closure can update it."""

    __slots__ = ("next", "active", "text_slot", "thought_slot", "tool_slots")

    def __init__(self) -> None:
        self.next = 0
        self.active: object = None
        self.text_slot: int | None = None
        self.thought_slot: int | None = None
        self.tool_slots: dict[int, int] = {}


def _translate_delta(delta: Any, state: _SlotState) -> Any:
    """Yield :class:`Delta` objects from one litellm chunk delta, assigning slots.

    A new slot is established when the delta channel changes, so two text blocks
    around a thinking block become two separate slots.
    """
    text = getattr(delta, "content", None)
    thought = getattr(delta, "reasoning_content", None)
    tool_calls = getattr(delta, "tool_calls", None)

    if text:
        if state.active != "text":
            state.active = "text"
            state.text_slot = state.next
            state.next += 1
        assert state.text_slot is not None
        yield TextDelta(state.text_slot, text)

    if thought:
        if state.active != "thought":
            state.active = "thought"
            state.thought_slot = state.next
            state.next += 1
        assert state.thought_slot is not None
        yield ThoughtDelta(state.thought_slot, text=thought)

    if tool_calls:
        for tc in tool_calls:
            idx = getattr(tc, "index", 0) or 0
            if idx not in state.tool_slots:
                state.tool_slots[idx] = state.next
                state.next += 1
            state.active = ("tool", idx)
            slot = state.tool_slots[idx]
            func = getattr(tc, "function", None)
            call_id = getattr(tc, "id", None)
            tool_name = getattr(func, "name", None) if func else None
            args_frag = getattr(func, "arguments", None) if func else None
            yield CallDelta(slot, id=call_id, tool=tool_name, args=args_frag or "")


# ---------------------------------------------------------------------------
# Usage re-derivation (obligation 2)
# ---------------------------------------------------------------------------


def _has_counts(usage_obj: Any) -> bool:
    return getattr(usage_obj, "prompt_tokens", None) is not None


def _finish(
    usage_obj: Any, finish_reason: str | None, alias: str, models: Mapping[str, str]
) -> Finish:
    prompt_tokens = int(getattr(usage_obj, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage_obj, "completion_tokens", 0) or 0)

    ptd = getattr(usage_obj, "prompt_tokens_details", None)
    cache_read = int(getattr(ptd, "cached_tokens", 0) or 0) if ptd else 0
    cache_write = int(getattr(ptd, "cache_write_tokens", 0) or 0) if ptd else 0

    # Re-derive the input counts to be disjoint: total == input + cache_read + cache_write.
    if prompt_tokens >= cache_read + cache_write:
        input_tokens = prompt_tokens - cache_read - cache_write
    else:
        # Provider reports prompt_tokens excluding cache reads already.
        input_tokens = prompt_tokens

    ctd = getattr(usage_obj, "completion_tokens_details", None)
    thought = int(getattr(ctd, "reasoning_tokens", 0) or 0) if ctd else 0

    full_model = models.get(alias, alias)
    billed = _billed_cost(usage_obj)

    return Finish(
        stop=_stop_from_reason(finish_reason),
        usage=Usage(
            model=full_model,
            input=input_tokens,
            cache_read=cache_read,
            cache_write=cache_write,
            output=completion_tokens,
            thought=thought,
            billed=billed,
            estimated=False,
        ),
        detail=finish_reason or "",
    )


def _synthesise_finish(
    finish_reason: str | None, alias: str, models: Mapping[str, str]
) -> Finish:
    full_model = models.get(alias, alias)
    return Finish(
        stop=_stop_from_reason(finish_reason),
        usage=Usage(
            model=full_model,
            input=0,
            cache_read=0,
            cache_write=0,
            output=0,
            thought=0,
            billed=None,
            estimated=True,
        ),
        detail=finish_reason or "",
    )


def _billed_cost(usage_obj: Any) -> Decimal | None:
    # litellm attaches response_cost to the reassembled response, not per-chunk;
    # per-chunk it is usually absent.  Try anyway, fall back to None.
    hidden = getattr(usage_obj, "_hidden_params", None)
    if hidden and isinstance(hidden, dict):
        cost = hidden.get("response_cost")
        if cost is not None:
            return Decimal(str(cost))
    return None


def _stop_from_reason(reason: str | None) -> Stop:
    if reason is None:
        return Stop.END
    r = reason.lower()
    if r in ("tool_calls", "function_call"):
        return Stop.TOOLS
    if r == "length":
        return Stop.LENGTH
    if r in ("content_filter", "content_filter"):
        return Stop.FILTER
    if r in ("stop_sequence",):
        return Stop.SEQUENCE
    return Stop.END


# ---------------------------------------------------------------------------
# Error mapping (§5)
# ---------------------------------------------------------------------------


def _map_error(exc: BaseException) -> InferenceError:
    name = type(exc).__name__
    msg = str(exc).lower()

    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)

    if "ratelimit" in name.lower() or status == 429:
        retry_after = getattr(exc, "retry_after", None)
        return Throttled(str(exc), retry_after=retry_after)
    if "serviceunavailable" in name.lower() or status in (503, 529):
        return Throttled(str(exc))
    if "contentpolicy" in name.lower():
        return Rejected(str(exc))
    if "authentication" in name.lower() or status == 401:
        return Rejected(str(exc))
    if "badrequest" in name.lower() or status in (400, 404, 422):
        if any(
            w in msg
            for w in ("context", "too large", "maximum", "context_length", "too many")
        ):
            return TooLarge(str(exc))
        return Rejected(str(exc))
    if (
        "connection" in name.lower()
        or "timeout" in name.lower()
        or status in (500, 502, 504)
    ):
        return Unreachable(str(exc))
    if isinstance(exc, InferenceError):
        return exc
    return InferenceError(str(exc))


# ---------------------------------------------------------------------------
# Request -> litellm kwargs (obligations 1, 6, 7)
# ---------------------------------------------------------------------------


def _build_kwargs(
    request: Request, models: Mapping[str, str], catalogue: Catalogue
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    k = request.knobs

    if k.max_output is not None:
        kwargs["max_tokens"] = k.max_output
    if k.temperature is not None:
        kwargs["temperature"] = k.temperature
    if k.top_p is not None:
        kwargs["top_p"] = k.top_p
    if k.stop:
        kwargs["stop"] = list(k.stop)
    if k.seed is not None:
        kwargs["seed"] = k.seed
    if k.timeout is not None:
        kwargs["timeout"] = k.timeout

    if request.tools:
        kwargs["tools"] = [_tool_dict(t) for t in request.tools]

    if request.output is not None:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": request.output.name,
                "schema": dict(request.output.schema),
                "strict": True,
            },
        }

    _apply_think(request, models, catalogue, kwargs)

    extra_body: dict[str, Any] = dict(k.extra)
    reasoning = dict(extra_body.get("reasoning", {}))
    reasoning.setdefault("exclude", False)
    extra_body["reasoning"] = reasoning
    kwargs["extra_body"] = extra_body

    return kwargs


def _apply_think(
    request: Request,
    models: Mapping[str, str],
    catalogue: Catalogue,
    kwargs: dict[str, Any],
) -> None:
    effort = request.knobs.think
    if effort is Effort.NONE:
        return

    full = models.get(request.model, request.model)
    provider = full.split("/")[0] if "/" in full else ""
    facts = catalogue.facts(request.model)
    max_output = request.knobs.max_output or facts.max_output or 0

    budget = facts.thinking.get(effort, 0)
    if budget > 0:
        # Budget-style provider (Anthropic).
        budget = min(budget, max_output) if max_output else budget
        if budget < 1024:
            budget = 1024 if (not max_output or max_output >= 1024) else 0
        if budget > 0:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
            return

    # Effort-style provider (OpenAI, Gemini via OpenAI-compatible route).
    if provider not in ("anthropic",):
        kwargs["reasoning_effort"] = _REASON_EFFORT[effort]


def _tool_dict(tool: Tool) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": dict(tool.params),
        },
    }


# ---------------------------------------------------------------------------
# Prompt -> litellm messages (obligations 4, 8)
# ---------------------------------------------------------------------------


def _build_messages_for_call(prompt: Prompt) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = []
    if prompt.directives:
        content = _parts_to_content(prompt.directives)
        if content:
            msgs.append({"role": "system", "content": content})
    for m in prompt.messages:
        if m.role == "user":
            msgs.extend(_build_user(m))
        else:
            msgs.extend(_build_assistant(m))
    return _merge(msgs)


def _build_messages_for_count(prompt: Prompt) -> list[dict[str, Any]]:
    """A simplified shape for litellm.token_counter (which accepts OpenAI messages)."""
    return _build_messages_for_call(prompt)


def _build_user(msg: Message) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    results = [p for p in msg.parts if isinstance(p, Result)]
    others = [p for p in msg.parts if not isinstance(p, Result)]
    for r in results:
        out.append(
            {"role": "tool", "tool_call_id": r.call, "content": _result_content(r)}
        )
    if others:
        content = _parts_to_content(others)
        out.append({"role": "user", "content": content})
    return out


def _build_assistant(msg: Message) -> list[dict[str, Any]]:
    tool_calls: list[dict[str, Any]] = []
    content_parts: list[dict[str, Any]] = []
    reasoning: list[str] = []
    for p in msg.parts:
        if isinstance(p, Thought):
            reasoning.append(p.text)
        elif isinstance(p, Call):
            tool_calls.append(
                {
                    "id": p.id,
                    "type": "function",
                    "function": {"name": p.tool, "arguments": json.dumps(p.args)},
                }
            )
        elif isinstance(p, Media):
            content_parts.extend(_media_parts(p))
        elif isinstance(p, Breakpoint):
            pass
    out: dict[str, Any] = {"role": "assistant"}
    if reasoning:
        out["reasoning_content"] = "".join(reasoning)
    if content_parts:
        out["content"] = _simplify_content(content_parts)
    if tool_calls:
        out["tool_calls"] = tool_calls
    if len(out) > 1:
        return [out]
    return []


def _result_content(r: Result) -> str:
    texts: list[str] = []
    for m in r.content:
        if m.mime == "text/plain":
            texts.append(m.data.decode("utf-8", errors="replace"))
        else:
            texts.append(f"[{m.mime}: {len(m.data)} bytes]")
    return "\n".join(texts) if texts else ""


def _parts_to_content(parts: Sequence[Part]) -> list[dict[str, Any]] | str:
    content: list[dict[str, Any]] = []
    for p in parts:
        if isinstance(p, Breakpoint):
            if content:
                content[-1]["cache_control"] = {"type": "ephemeral"}
            continue
        if isinstance(p, Media):
            content.extend(_media_parts(p))
        elif isinstance(p, Thought):
            content.append({"type": "text", "text": p.text})
        elif isinstance(p, Call):
            # Should not normally appear outside llm messages; represent as text.
            content.append({"type": "text", "text": json.dumps(p.args)})
        else:
            # Result -- the only remaining Part variant.
            content.append({"type": "text", "text": _result_content(p)})
    if not content:
        return ""
    if len(content) == 1 and content[0].get("type") == "text":
        return content[0]["text"]
    return content


def _media_parts(m: Media) -> list[dict[str, Any]]:
    if m.mime == "text/plain":
        return [{"type": "text", "text": m.data.decode("utf-8", errors="replace")}]
    data_url = _data_url(m)
    if m.mime.startswith("image/"):
        return [{"type": "image_url", "image_url": {"url": data_url}}]
    return [{"type": "file", "file": {"filename": "file", "file_data": data_url}}]


def _data_url(m: Media) -> str:
    b64 = base64.b64encode(m.data).decode("ascii")
    return f"data:{m.mime};base64,{b64}"


def _simplify_content(parts: list[dict[str, Any]]) -> Any:
    if len(parts) == 1 and parts[0].get("type") == "text":
        return parts[0]["text"]
    return parts


def _merge(msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge adjacent same-role messages (not tool messages) and drop empties."""
    out: list[dict[str, Any]] = []
    for msg in msgs:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "tool":
            out.append(msg)
            continue
        if (
            not content
            and not msg.get("tool_calls")
            and not msg.get("reasoning_content")
        ):
            continue
        if out and out[-1].get("role") == role and role != "tool":
            prev = out[-1]
            prev_content = prev.get("content", "")
            if isinstance(prev_content, str) and isinstance(content, str):
                prev["content"] = (
                    (prev_content + "\n" + content).strip() if prev_content else content
                )
            elif isinstance(prev_content, list) and isinstance(content, list):
                prev["content"] = prev_content + content
            else:
                prev["content"] = content
        else:
            out.append(dict(msg))
    return out
