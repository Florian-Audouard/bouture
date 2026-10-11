"""Données d'un import d'historique existant (D104)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal
from uuid import UUID


@dataclass
class ImportedEntry:
    """Une entrée importée, rattachée à ``parent_id`` (``None`` : la racine de la session).

    - ``user`` / ``assistant`` : ``content`` est une chaîne ou une liste de blocs ;
    - ``state_edit`` : ``state`` est l'état complet après l'édition, le patch est calculé.

    Les entrées se suivent dans l'ordre de la liste : un parent précède ses enfants.
    ``created_at`` est conservé et ordonne les branches sœurs.
    """

    id: UUID
    parent_id: UUID | None
    type: Literal["user", "assistant", "state_edit"]
    created_at: datetime
    content: Any = None
    state: Any = None
    checkpoint: bool = True
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict[str, Any])


@dataclass
class ImportedPublication:
    """Version publiée existante : l'état de ``entry_id`` (un checkpoint importé)."""

    entry_id: UUID
    version: int
    published_at: datetime
