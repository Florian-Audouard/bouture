"""Références vers des objets publiés et inputs de session (D50, D54)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID


@dataclass(frozen=True, slots=True)
class Ref:
    """Référence vers un objet publié. ``version=None`` → dernière version, figée à la création."""

    kind: str
    id: UUID
    version: int | None = None


@dataclass(frozen=True, slots=True)
class Data:
    """Snapshot de données passé en input. ``value`` est validé contre le type déclaré du kind."""

    kind: str
    value: Any


@dataclass(frozen=True, slots=True)
class PublishedRef:
    kind: str
    object_id: UUID
    version: int

    def as_ref(self) -> Ref:
        return Ref(self.kind, self.object_id, self.version)


Input = Ref | Data
