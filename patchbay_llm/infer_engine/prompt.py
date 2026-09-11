"""The prompt.

This is the important type.  It is the output of ``render`` -- the projection
of state into something a model can be shown -- and therefore the thing
Sections compose and ``compare`` diffs.

The shapes here are specified in INFERENCE.md §2.  Nothing provider-shaped
(OpenAI field names, litellm types) appears anywhere in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal, Mapping, Union

__all__ = [
    "Role",
    "Thought",
    "Media",
    "Call",
    "Result",
    "Breakpoint",
    "Part",
    "Message",
    "Prompt",
    "Problem",
    "check",
]

Role = Literal["user", "llm"]


@dataclass(frozen=True)
class Thought:
    """A reasoning block previously emitted by the model.

    ``signature`` is an opaque provider token that must be replayed verbatim
    when the thought is echoed back in a follow-up request (Anthropic extended
    thinking + tool use).
    """

    text: str
    signature: str | None = None


@dataclass(frozen=True)
class Media:
    """A content part: text, image, audio or PDF, distinguished only by mime.

    ``data`` is always explicit bytes.
    """

    mime: str
    data: bytes


@dataclass(frozen=True)
class Call:
    """A tool call the model made.  ``args`` is parsed, not a JSON string."""

    id: str
    tool: str
    args: Mapping[str, Any]


@dataclass(frozen=True)
class Result:
    """The answer to a :class:`Call`.  ``call`` is the ``Call.id`` it answers."""

    call: str
    content: tuple[Media, ...]
    failed: bool = False


@dataclass(frozen=True)
class Breakpoint:
    """Cache everything up to and including the part before me.

    A cache breakpoint is a claim about a *prefix*, so making it a marker in the
    sequence (rather than a field on content) means the cached prefix is a
    slice and content stays content.  Breakpoints are advisory to the provider.
    """


Part = Union[Thought, Media, Call, Result, Breakpoint]


@dataclass(frozen=True)
class Message:
    """One message in a prompt: a role and an ordered tuple of parts."""

    role: Role
    parts: tuple[Part, ...]


@dataclass(frozen=True)
class Prompt:
    """A finished artefact with directives fixed and breakpoints placed.

    Only ``assemble`` may produce one.  ``extend`` is the only append that
    happens in practice -- it appends messages only and cannot touch
    directives, so it is prefix-preserving by construction.
    """

    directives: tuple[Part, ...] = ()
    messages: tuple[Message, ...] = ()

    def extend(self, *messages: Message) -> Prompt:
        """Append messages; directives are untouched (prefix-preserving)."""
        return replace(self, messages=self.messages + messages)


@dataclass(frozen=True)
class Problem:
    """A validation problem found by :func:`check`."""

    message: str


def check(prompt: Prompt) -> tuple[Problem, ...]:
    """Validate a prompt.  An empty result means the prompt is valid.

    The invariants that matter are referential (which the type system cannot
    express), so there is one mechanism instead of splitting ``Part`` into
    ``UserPart`` / ``LlmPart``.
    """
    problems: list[Problem] = []

    # Directives may not contain Call/Result -- those are conversational.
    for part in prompt.directives:
        if isinstance(part, Call | Result):
            problems.append(
                Problem(f"directives must not contain {type(part).__name__}")
            )

    seen_calls: set[str] = set()
    pending_calls: set[str] = set()

    for msg in prompt.messages:
        if msg.role == "user":
            for part in msg.parts:
                if isinstance(part, Thought):
                    problems.append(Problem("Thought may only appear in llm messages"))
                if isinstance(part, Call):
                    problems.append(Problem("Call may only appear in llm messages"))
                if isinstance(part, Result):
                    if part.call not in seen_calls:
                        problems.append(
                            Problem(f"Result references unknown Call id {part.call!r}")
                        )
                    pending_calls.discard(part.call)
        else:  # llm
            # Every Call from the previous llm message must be answered before
            # this one arrives.
            if pending_calls:
                problems.append(
                    Problem(
                        f"unanswered Call(s) before next llm message: {sorted(pending_calls)}"
                    )
                )
                pending_calls.clear()
            for part in msg.parts:
                if isinstance(part, Result):
                    problems.append(Problem("Result may only appear in user messages"))
                if isinstance(part, Call):
                    seen_calls.add(part.id)
                    pending_calls.add(part.id)

    # Trailing unanswered Calls (the model just called tools) are normal -- the
    # round is not finished yet -- so they are not flagged here.
    return tuple(problems)
