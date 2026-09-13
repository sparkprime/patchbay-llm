"""The prompt.

This is the important type.  It is the output of ``render`` -- the projection
of state into something a model can be shown -- and therefore the thing
Sections compose and ``compare`` diffs.

Nothing provider-shaped (OpenAI field names, litellm types) appears anywhere in
this module.
"""

from dataclasses import dataclass, replace
from typing import Any, Literal, Mapping, Self, Union

from patchbay_llm.events import Media

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

    def extend(self, *messages: Message) -> Self:
        """Append messages; directives are untouched (prefix-preserving)."""
        return replace(self, messages=self.messages + messages)
