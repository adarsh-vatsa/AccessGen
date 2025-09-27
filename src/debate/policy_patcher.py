#!/usr/bin/env python3
"""Minimal JSON Patch applier for debate pipeline."""
from __future__ import annotations

import copy
from typing import Any, Dict, List


class PatchError(Exception):
    pass


def _resolve_path(obj: Any, path: str, create: bool = False) -> Any:
    """Navigate JSON Pointer path, returning parent container for final segment."""
    if not path.startswith("/") and path != "":
        raise PatchError(f"Unsupported path: {path}")
    parts = [p for p in path.split("/") if p]
    cur = obj
    for idx, part in enumerate(parts[:-1]):
        if isinstance(cur, list):
            if part == "-":
                raise PatchError("'-' not allowed in intermediate path elements")
            try:
                index = int(part)
            except ValueError as exc:
                raise PatchError(f"Invalid list index: {part}") from exc
            if index >= len(cur):
                if create:
                    cur.extend({} for _ in range(index - len(cur) + 1))
                else:
                    raise PatchError(f"Index out of bounds: {part}")
            cur = cur[index]
        elif isinstance(cur, dict):
            if part not in cur:
                if create:
                    cur[part] = {}
                else:
                    raise PatchError(f"Missing key: {part}")
            cur = cur[part]
        else:
            raise PatchError("Cannot traverse non-container")
    return cur, (parts[-1] if parts else None)


def apply_patches(policy: Dict[str, Any], patches: List[Dict[str, Any]]) -> Dict[str, Any]:
    result = copy.deepcopy(policy)
    for patch in patches:
        op = patch.get("op")
        path = patch.get("path", "")
        if op not in {"add", "replace", "remove"}:
            raise PatchError(f"Unsupported op: {op}")

        parent, key = _resolve_path(result, path, create=(op == "add"))

        if isinstance(parent, list):
            if key is None:
                raise PatchError("List operation requires index")
            if key == "-":
                index = len(parent)
            else:
                try:
                    index = int(key)
                except ValueError as exc:
                    raise PatchError(f"Invalid list index: {key}") from exc
            if op == "remove":
                if index >= len(parent):
                    raise PatchError("Remove index out of range")
                parent.pop(index)
            elif op == "add":
                value = patch.get("value")
                if index > len(parent):
                    raise PatchError("Add index out of range")
                parent.insert(index, value)
            else:  # replace
                if index >= len(parent):
                    raise PatchError("Replace index out of range")
                parent[index] = patch.get("value")
        elif isinstance(parent, dict):
            if key is None:
                raise PatchError("Object operation requires key")
            if op == "remove":
                if key not in parent:
                    raise PatchError(f"Key '{key}' not present for remove")
                del parent[key]
            elif op == "add" or op == "replace":
                parent[key] = patch.get("value")
        else:
            raise PatchError("Cannot apply patch to primitive value")

    return result


