"""Navigation dans l'arbre d'entrées d'une session."""

from __future__ import annotations

from collections.abc import Iterable
from uuid import UUID

from .errors import InvalidTarget, NotFound
from .storage.base import EntryRecord


class Tree:
    def __init__(self, entries: Iterable[EntryRecord]) -> None:
        self.by_id: dict[UUID, EntryRecord] = {}
        # ``children`` ignore les branches à la corbeille (D100) ; ``all_children`` non.
        self.children: dict[UUID | None, list[EntryRecord]] = {}
        self.all_children: dict[UUID | None, list[EntryRecord]] = {}
        # Ordre chronologique, l'id départage (D104 : une entrée importée garde sa date).
        for e in sorted(entries, key=lambda e: (e.created_at, e.id)):
            self.by_id[e.id] = e
            self.all_children.setdefault(e.parent_id, []).append(e)
            if e.hidden_at is None:
                self.children.setdefault(e.parent_id, []).append(e)

    def get(self, entry_id: UUID) -> EntryRecord:
        try:
            return self.by_id[entry_id]
        except KeyError:
            raise NotFound(f"entrée {entry_id} introuvable") from None

    @property
    def root(self) -> EntryRecord:
        return self.children[None][0]

    def path(self, entry_id: UUID) -> list[EntryRecord]:
        """Chemin root → entry (inclus)."""
        out: list[EntryRecord] = []
        cur: UUID | None = entry_id
        while cur is not None:
            e = self.get(cur)
            out.append(e)
            cur = e.parent_id
        out.reverse()
        return out

    def siblings(self, entry_id: UUID) -> tuple[list[EntryRecord], int]:
        """Frères visibles ; l'entrée elle-même y figure même si elle est à la corbeille."""
        e = self.get(entry_id)
        sibs = [
            s
            for s in self.all_children.get(e.parent_id, [])
            if s.hidden_at is None or s.id == entry_id
        ]
        return sibs, next(i for i, s in enumerate(sibs) if s.id == entry_id)

    def subtree(self, entry_id: UUID) -> list[EntryRecord]:
        """L'entrée et tous ses descendants, corbeille comprise."""
        out = [self.get(entry_id)]
        i = 0
        while i < len(out):
            out += self.all_children.get(out[i].id, [])
            i += 1
        return out

    def is_hidden(self, entry_id: UUID) -> bool:
        return any(e.hidden_at is not None for e in self.path(entry_id))

    def latest_descendant(self, entry_id: UUID) -> EntryRecord:
        """Descend en suivant l'enfant le plus récent à chaque niveau (D16)."""
        cur = self.get(entry_id)
        while kids := self.children.get(cur.id):
            cur = kids[-1]
        return cur

    def nearest_checkpoint(self, entry_id: UUID) -> EntryRecord:
        for e in reversed(self.path(entry_id)):
            if e.is_checkpoint:
                return e
        return self.root

    def turn_user(self, entry_id: UUID) -> EntryRecord:
        """Message ``user`` qui a ouvert le tour contenant ``entry_id`` (D15)."""
        for e in reversed(self.path(entry_id)):
            if e.type == "user":
                return e
        raise InvalidTarget("aucun message utilisateur avant cette entrée")
