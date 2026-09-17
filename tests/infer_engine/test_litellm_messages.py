"""Tests for the litellm message-building helpers (prompt -> provider messages).

Focus: multimodal ``Result`` content is preserved as real content parts
(``image_url`` / ``file``) rather than the old lossy ``[mime: N bytes]``
placeholder, while all-text results keep their historical string shape.
"""

# pyright: reportPrivateUsage=false

from typing import Any

from patchbay_llm.events import Media
from patchbay_llm.infer_engine.litellm import (
    _build_messages_for_call,
    _build_user,
    _parts_to_content,
    _result_content,
)
from patchbay_llm.infer_engine.prompt import Message, Prompt, Result


def _png(b: bytes = b"\x89PNG\r\n") -> Media:
    return Media("image/png", b)


def _image_url_parts(content: "str | list[dict[str, Any]]") -> list[dict[str, Any]]:
    assert isinstance(content, list)
    return content


# --- _result_content -----------------------------------------------------


def test_result_all_text_returns_joined_string() -> None:
    """All-text results collapse to a single newline-joined string (no regression)."""
    r = Result(
        call="c1",
        content=(Media("text/plain", b"hello"), Media("text/plain", b"world")),
    )
    assert _result_content(r) == "hello\nworld"


def test_result_single_text_returns_plain_string() -> None:
    """A single text media item returns its plain string value."""
    r = Result(call="c1", content=(Media("text/plain", b"18C, cloudy"),))
    assert _result_content(r) == "18C, cloudy"


def test_result_empty_content_returns_empty_string() -> None:
    """A result with no content parts yields an empty string."""
    assert _result_content(Result(call="c1", content=())) == ""


def test_result_single_image_returns_image_url_part() -> None:
    """A single image media item becomes an ``image_url`` content part."""
    r = Result(call="c1", content=(_png(b"\x89PNG"),))
    out = _image_url_parts(_result_content(r))
    assert len(out) == 1
    part: dict[str, Any] = out[0]
    assert part["type"] == "image_url"
    url: str = part["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")


def test_result_mixed_text_and_image_returns_both_parts() -> None:
    """Mixed text and image content produces both ``text`` and ``image_url`` parts."""
    r = Result(
        call="c1",
        content=(Media("text/plain", b"here is the screenshot:"), _png(b"\x89PNG")),
    )
    out = _image_url_parts(_result_content(r))
    assert len(out) == 2
    assert out[0] == {"type": "text", "text": "here is the screenshot:"}
    assert out[1]["type"] == "image_url"


def test_result_non_image_non_text_uses_file_part() -> None:
    """Non-image, non-text content is emitted as a ``file`` content part."""
    r = Result(call="c1", content=(Media("application/pdf", b"%PDF-1.4"),))
    out = _image_url_parts(_result_content(r))
    assert out[0]["type"] == "file"
    assert out[0]["file"]["file_data"].startswith("data:application/pdf;base64,")


# --- _build_user ---------------------------------------------------------


def test_build_user_image_result_emits_tool_message_with_image_content() -> None:
    """A tool result carrying an image must reach the provider as real image content."""
    msg = Message(
        role="user", parts=(Result(call="call_7", content=(_png(),), failed=False),)
    )
    out = _build_user(msg)
    assert len(out) == 1
    tool_msg: dict[str, Any] = out[0]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "call_7"
    content = tool_msg["content"]
    parts = _image_url_parts(content)
    assert any(p.get("type") == "image_url" for p in parts)


def test_build_user_text_result_still_uses_plain_string() -> None:
    """A text-only tool result keeps the historical plain-string content shape."""
    msg = Message(
        role="user",
        parts=(
            Result(call="call_7", content=(Media("text/plain", b"ok"),), failed=False),
        ),
    )
    out = _build_user(msg)
    assert len(out) == 1
    assert out[0]["content"] == "ok"


def test_build_user_no_lossy_placeholder_for_images() -> None:
    """Regression guard: no ``[image/png: N bytes]`` placeholder should appear."""
    msg = Message(
        role="user",
        parts=(Result(call="c1", content=(_png(b"\x89PNG\r\n\x1a\n"),), failed=False),),
    )
    out = _build_user(msg)
    content = out[0]["content"]
    assert not (isinstance(content, str) and content.startswith("[image/png:"))


# --- _parts_to_content (Result mixed with other user parts) -------------


def test_parts_to_content_result_image_not_double_nested() -> None:
    """A Result inside a user (non-tool) part list must flatten its image parts."""
    parts = (
        Media("text/plain", b"prompt"),
        Result(call="c1", content=(_png(),), failed=False),
    )
    out = _parts_to_content(parts)
    assert isinstance(out, list)
    types = [p.get("type") for p in out]
    assert "text" in types and "image_url" in types
    for p in out:
        if p.get("type") == "text":
            assert not p["text"].startswith("[")


def test_parts_to_content_all_text_result_collapses_to_string() -> None:
    """An all-text Result in a user part list collapses to a plain string."""
    parts = (
        Result(call="c1", content=(Media("text/plain", b"all text"),), failed=False),
    )
    assert _parts_to_content(parts) == "all text"


# --- end-to-end via _build_messages_for_call -----------------------------


def test_build_messages_for_call_routes_image_tool_result_as_list_content() -> None:
    """End-to-end build routes an image tool result as list content, not a string."""
    msg = Message(
        role="user", parts=(Result(call="c1", content=(_png(),), failed=False),)
    )
    prompt = Prompt(directives=(), messages=(msg,))
    built = _build_messages_for_call(prompt)
    tool_msgs = [m for m in built if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    content = tool_msgs[0]["content"]
    parts = _image_url_parts(content)
    assert parts[0]["type"] == "image_url"
