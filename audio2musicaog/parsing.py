"""Robust extraction of structured answers from model text."""

from __future__ import annotations

import json
from typing import Any, Iterable


class JsonExtractionError(ValueError):
    """Raised when no suitable JSON object can be recovered from an answer."""


def _candidate_objects(text: str) -> Iterable[dict[str, Any]]:
    """Yield JSON objects embedded in arbitrary model output.

    Thinking models may emit prose, ``<think>`` blocks, Markdown fences, or
    multiple example objects before their final answer.  ``raw_decode`` lets us
    recover balanced JSON without using a fragile first-``{``/last-``}`` slice.
    """

    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            yield value


def extract_json_object(text: str, required_keys: Iterable[str] = ()) -> dict[str, Any]:
    """Return the last embedded JSON object containing all ``required_keys``."""

    required = set(required_keys)
    matches = [item for item in _candidate_objects(text) if required.issubset(item)]
    if not matches:
        detail = f" containing keys {sorted(required)}" if required else ""
        raise JsonExtractionError(f"No valid JSON object{detail} found in model response")
    return matches[-1]

