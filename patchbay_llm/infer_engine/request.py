"""The request.

A frozen struct, not keyword arguments -- it is what gets logged, validation
has a home, and ``compare`` can vary one field via ``replace``.

``tools`` and ``output`` are exclusive: combining them is unportable and
non-deterministic in reply shape, so the constructor refuses it.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Mapping

from patchbay_llm.infer_engine.prompt import Prompt
from patchbay_llm.infer_engine.schema import normalise

__all__ = ["Effort", "Knobs", "Tool", "Schema", "Request"]


class Effort(Enum):
    """Reasoning effort, four levels.  No token budget.

    The enum normalises onto all three provider dialects; a token budget
    normalises onto none of them.
    """

    NONE = auto()
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()


@dataclass(frozen=True)
class Knobs:
    """Inference parameters.  ``extra`` is unportable, on purpose."""

    max_output: int | None = None
    think: Effort = Effort.NONE
    temperature: float | None = None
    top_p: float | None = None
    stop: tuple[str, ...] = ()
    seed: int | None = None
    timeout: float | None = None
    extra: Mapping[str, Any] = field(default_factory=dict[str, Any])


@dataclass(frozen=True)
class Tool:
    """A tool the model may call.  ``params`` is a JSON Schema, normalised."""

    name: str
    description: str
    params: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", normalise(self.params))


@dataclass(frozen=True)
class Schema:
    """A structured-output schema.  Normalised at construction."""

    schema: Mapping[str, Any]
    name: str = "output"

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema", normalise(self.schema))


@dataclass(frozen=True)
class Request:
    """One inference request.  ``tools`` and ``output`` are exclusive."""

    model: str
    prompt: Prompt
    tools: tuple[Tool, ...] = ()
    output: Schema | None = None
    knobs: Knobs = Knobs()

    def __post_init__(self) -> None:
        if self.tools and self.output:
            raise ValueError("tools and output are exclusive")
