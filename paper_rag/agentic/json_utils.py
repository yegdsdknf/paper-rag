from __future__ import annotations

import json
import re
from typing import Any


_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def parse_json_object(text: str) -> dict[str, Any]:
    candidates = [text.strip()]
    candidates.extend(match.strip() for match in _FENCED_JSON_RE.findall(text))

    candidates.extend(_extract_json_objects(text))

    last_error: Exception | None = None
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if not isinstance(parsed, dict):
            last_error = ValueError("Expected a JSON object.")
            continue
        return parsed

    raise ValueError("Could not parse a JSON object.") from last_error


def _extract_json_objects(text: str) -> list[str]:
    decoder = json.JSONDecoder()
    objects: list[str] = []
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            objects.append(text[start : start + end])
    return objects
