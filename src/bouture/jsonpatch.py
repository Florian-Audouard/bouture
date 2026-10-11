"""JSON Patch (RFC 6902) minimal : diff et application (opérations add / remove / replace)."""

from __future__ import annotations

import copy
from typing import Any

Patch = list[dict[str, Any]]


def _escape(key: str) -> str:
    return key.replace("~", "~0").replace("/", "~1")


def _unescape(token: str) -> str:
    return token.replace("~1", "/").replace("~0", "~")


def diff(old: Any, new: Any, path: str = "") -> Patch:
    if old == new:
        return []
    if isinstance(old, dict) and isinstance(new, dict):
        ops: Patch = []
        for k in old:
            if k not in new:
                ops.append({"op": "remove", "path": f"{path}/{_escape(str(k))}"})
        for k, v in new.items():
            p = f"{path}/{_escape(str(k))}"
            if k not in old:
                ops.append({"op": "add", "path": p, "value": copy.deepcopy(v)})
            else:
                ops.extend(diff(old[k], v, p))
        return ops
    if isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        ops = []
        for i, (a, b) in enumerate(zip(old, new, strict=True)):
            ops.extend(diff(a, b, f"{path}/{i}"))
        return ops
    return [{"op": "replace", "path": path, "value": copy.deepcopy(new)}]


def apply(doc: Any, patch: Patch) -> Any:
    doc = copy.deepcopy(doc)
    for op in patch:
        tokens = [_unescape(t) for t in op["path"].split("/")[1:]] if op["path"] else []
        if not tokens:
            if op["op"] in ("add", "replace"):
                doc = copy.deepcopy(op["value"])
                continue
            raise ValueError("remove sur la racine")
        parent = doc
        for t in tokens[:-1]:
            parent = parent[int(t)] if isinstance(parent, list) else parent[t]
        last = tokens[-1]
        if isinstance(parent, list):
            idx = len(parent) if last == "-" else int(last)
            if op["op"] == "add":
                parent.insert(idx, copy.deepcopy(op["value"]))
            elif op["op"] == "remove":
                del parent[idx]
            else:
                parent[idx] = copy.deepcopy(op["value"])
        else:
            if op["op"] == "remove":
                del parent[last]
            else:
                parent[last] = copy.deepcopy(op["value"])
    return doc
