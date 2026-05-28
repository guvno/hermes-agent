"""Shared helpers for classifying tool result payloads."""

from __future__ import annotations

import json
from typing import Any


FILE_MUTATING_TOOL_NAMES = frozenset({"write_file", "patch"})


def _load_leading_json_object(result: str) -> Any:
    """Parse a JSON object from the start of a tool result string.

    File-tool results are JSON, but runtime layers may append non-JSON
    guidance after the payload, e.g. a tool-loop warning.  Strict
    ``json.loads`` then fails even though the leading JSON proves whether
    the mutation landed.  ``raw_decode`` preserves strictness for the
    payload itself while tolerating well-known trailing advisory text.
    """
    stripped = result.strip()
    if not stripped.startswith("{"):
        return None
    decoder = json.JSONDecoder()
    try:
        data, _end = decoder.raw_decode(stripped)
    except Exception:
        return None
    return data


def file_mutation_result_landed(tool_name: str, result: Any) -> bool:
    """Return True when a file mutation result proves the write landed."""
    if tool_name not in FILE_MUTATING_TOOL_NAMES or not isinstance(result, str):
        return False
    data = _load_leading_json_object(result)
    if not isinstance(data, dict) or data.get("error"):
        return False
    if tool_name == "write_file":
        return "bytes_written" in data
    if tool_name == "patch":
        return data.get("success") is True
    return False
