"""The inference API.

One call. Everything patchbay-llm asks of an LLM goes through it::

    async for delta in engine.run(request):
        ...

Nothing provider-shaped (litellm types, OpenAI field names) escapes this
package.  See ``INFERENCE.md`` for the full specification.
"""

from __future__ import annotations

from patchbay_llm.infer_engine.assemble import (
    CacheHint,
    Contribution,
    assemble,
    place_breakpoints,
)
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
from patchbay_llm.infer_engine.engine import InferEngine
from patchbay_llm.infer_engine.errors import (
    InferenceError,
    Rejected,
    Throttled,
    TooLarge,
    Unreachable,
)
from patchbay_llm.infer_engine.litellm import LitellmCatalogue, LitellmInferEngine
from patchbay_llm.infer_engine.prompt import (
    Breakpoint,
    Call,
    Media,
    Message,
    Part,
    Problem,
    Prompt,
    Result,
    Thought,
    check,
)
from patchbay_llm.infer_engine.reply import Reply, collect
from patchbay_llm.infer_engine.request import Effort, Knobs, Request, Schema, Tool
from patchbay_llm.infer_engine.schema import Unsupported, normalise
from patchbay_llm.infer_engine.testing import check_stream

__all__ = [
    # prompt
    "Breakpoint",
    "Call",
    "Media",
    "Message",
    "Part",
    "Problem",
    "Prompt",
    "Result",
    "Thought",
    "check",
    # assemble
    "CacheHint",
    "Contribution",
    "assemble",
    "place_breakpoints",
    # request
    "Effort",
    "Knobs",
    "Request",
    "Schema",
    "Tool",
    # schema
    "Unsupported",
    "normalise",
    # delta
    "CallDelta",
    "Delta",
    "Finish",
    "Stop",
    "TextDelta",
    "ThoughtDelta",
    "Usage",
    # reply
    "Reply",
    "collect",
    # errors
    "InferenceError",
    "Rejected",
    "Throttled",
    "TooLarge",
    "Unreachable",
    # engine + catalogue
    "InferEngine",
    "Catalogue",
    "Facts",
    "Rate",
    "LitellmInferEngine",
    "LitellmCatalogue",
    # testing
    "check_stream",
]
