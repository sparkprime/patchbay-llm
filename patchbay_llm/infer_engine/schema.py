"""JSON Schema normalisation.

The supported JSON Schema subset is part of this API's contract: a schema that
works on one provider and not another breaks the one-file-swap claim.
``normalise`` rewrites schemas into the portable subset and **rejects loudly**
anything outside it -- never stripped, because a dropped ``pattern`` means the
schema you validated against is not the schema the model was constrained to.

Supported (after rewrite):
    - ``type`` of object/array/string/integer/number/boolean/null
    - ``properties``, ``required``, ``additionalProperties: false``
    - a single ``items`` schema
    - ``enum``, ``anyOf``, ``description`` anywhere

Rewritten:
    - ``$ref`` / ``$defs`` are inlined (cycles refused)
    - every object gets ``additionalProperties: false``
    - every property is added to ``required``; optional ones become
      ``anyOf: [T, {"type": "null"}]``

Rejected: oneOf, allOf, not, patternProperties, additionalProperties as a
schema, tuple-form items, minimum/maximum/minLength/maxLength/minItems/
maxItems/pattern/format, const, default, and recursive $ref.
"""

from typing import Any, Mapping

# JSON Schema handling is inherently dynamic (values are arbitrary JSON); the
# unknown-type noise from dict()/Mapping.get() carries no real information.
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false


__all__ = ["Unsupported", "normalise"]


class Unsupported(Exception):
    """Raised when a schema uses keywords outside the portable subset."""

    def __init__(self, keywords: list[str]) -> None:
        super().__init__(f"unsupported JSON Schema keywords: {keywords}")
        self.keywords = keywords


_REJECTED: set[str] = {
    "oneOf",
    "allOf",
    "not",
    "patternProperties",
    "minimum",
    "maximum",
    "minLength",
    "maxLength",
    "minItems",
    "maxItems",
    "pattern",
    "format",
    "const",
    "default",
    "multipleOf",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "minProperties",
    "maxProperties",
    "uniqueItems",
}


def _check_keys(schema: Mapping[str, Any]) -> None:
    bad = [k for k in schema if k in _REJECTED]
    if bad:
        raise Unsupported(bad)


def _inline(
    node: Mapping[str, Any], defs: Mapping[str, Any], stack: tuple[str, ...]
) -> dict[str, Any]:
    """Return a deep copy of ``node`` with every ``$ref`` resolved against defs.

    Recursive references (cycles) are refused -- a cycle cannot be inlined.
    """
    if "$ref" in node:
        ref = node["$ref"]
        if not (ref.startswith("#/$defs/") or ref.startswith("#/definitions/")):
            raise Unsupported([f"$ref {ref}"])
        name = ref.split("/")[-1]
        if name in stack:
            raise Unsupported([f"recursive $ref {ref}"])
        target = defs.get(name)
        if target is None:
            raise Unsupported([f"unknown $ref {ref}"])
        resolved = _inline(dict(target), defs, stack + (name,))
        siblings = {
            k: (_inline(v, defs, stack) if isinstance(v, dict) else v)
            for k, v in node.items()
            if k != "$ref"
        }
        return {**resolved, **siblings}

    out: dict[str, Any] = {}
    for key, value in node.items():
        if key in ("$defs", "definitions"):
            continue
        if isinstance(value, dict):
            out[key] = _inline(value, defs, stack)
        elif isinstance(value, list):
            out[key] = [
                _inline(x, defs, stack) if isinstance(x, dict) else x for x in value
            ]
        else:
            out[key] = value
    return out


def _rewrite(node: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the portable rewrites and enforce the subset."""
    _check_keys(node)

    out: dict[str, Any] = {
        k: v for k, v in node.items() if k not in ("$defs", "definitions")
    }

    if "anyOf" in out:
        out["anyOf"] = [_rewrite(x) if isinstance(x, dict) else x for x in out["anyOf"]]

    type_ = out.get("type")

    if type_ == "object":
        props = out.get("properties")
        if props is not None:
            if not isinstance(props, dict):
                raise Unsupported(["properties"])
            existing_required = set(out.get("required", []))
            new_props: dict[str, Any] = {}
            new_required: list[str] = []
            for name, sub in props.items():
                rewritten = _rewrite(sub) if isinstance(sub, dict) else sub
                if name in existing_required:
                    new_props[name] = rewritten
                else:
                    new_props[name] = {"anyOf": [rewritten, {"type": "null"}]}
                new_required.append(name)
            out["properties"] = new_props
            out["required"] = new_required

        ap = out.get("additionalProperties", False)
        if isinstance(ap, dict):
            raise Unsupported(["additionalProperties (as a schema)"])
        out["additionalProperties"] = False

    elif type_ == "array":
        if "items" in out:
            items = out["items"]
            if isinstance(items, list):
                raise Unsupported(["items (tuple form)"])
            out["items"] = _rewrite(items) if isinstance(items, dict) else items

    return out


def normalise(schema: Mapping[str, Any]) -> Mapping[str, Any]:
    """Normalise a JSON Schema into the portable subset.

    Raises :class:`Unsupported` if any keyword outside the subset is present.
    """
    defs = schema.get("$defs") or schema.get("definitions") or {}
    if not isinstance(defs, dict):
        raise Unsupported(["$defs"])
    inlined = _inline(dict(schema), defs, ())
    return _rewrite(inlined)
