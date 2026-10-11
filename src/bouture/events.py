"""Journal d'événements d'un tour (D24) et réveil local des abonnés."""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from .storage.base import EventRecord, NewEvent, Storage
from .views import Event

DELTA_TYPES = ("text_delta", "reasoning_delta", "tool_call_delta")


def now() -> datetime:
    return datetime.now(UTC)


def to_event(rec: EventRecord) -> Event:
    return Event(
        id=rec.id,
        session_id=rec.session_id,
        turn_id=rec.turn_id,
        type=rec.type,
        data=rec.data,
        ts=rec.created_at,
    )


class Hub:
    """Un ``asyncio.Event`` par session, réveillé à chaque écriture d'événements (locale ou LISTEN)."""

    def __init__(self) -> None:
        self._waiters: dict[UUID, asyncio.Event] = {}

    def waiter(self, session_id: UUID) -> asyncio.Event:
        ev = self._waiters.get(session_id)
        if ev is None:
            ev = self._waiters[session_id] = asyncio.Event()
        return ev

    def wake(self, session_id: UUID) -> None:
        ev = self._waiters.pop(session_id, None)
        if ev is not None:
            ev.set()


class EventSink:
    """Écrit les événements d'un tour dans ``turn_events``.

    Les deltas consécutifs d'une même entrée sont fusionnés et vidés toutes les
    ``flush_interval`` secondes ou dès ``flush_chars`` caractères ; tout autre événement
    vide d'abord le tampon, ce qui garantit l'ordre.
    """

    def __init__(
        self,
        storage: Storage,
        hub: Hub,
        session_id: UUID,
        turn_id: UUID,
        *,
        flush_interval: float = 0.1,
        flush_chars: int = 200,
    ) -> None:
        self._storage = storage
        self._hub = hub
        self.session_id = session_id
        self.turn_id = turn_id
        self._interval = flush_interval
        self._chars = flush_chars
        self._buf: list[tuple[str, dict[str, Any]]] = []
        self._buf_len = 0
        self._lock = asyncio.Lock()
        self._ticker: asyncio.Task[None] | None = None

    def start(self) -> None:
        self._ticker = asyncio.create_task(self._tick())

    async def close(self) -> None:
        if self._ticker is not None:
            self._ticker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._ticker
            self._ticker = None
        await self.flush()

    async def _tick(self) -> None:
        while True:
            await asyncio.sleep(self._interval)
            if self._buf:
                await self.flush()

    def new(self, type: str, data: dict[str, Any]) -> NewEvent:
        return NewEvent(self.session_id, self.turn_id, type, data, now())

    async def delta(self, type: str, entry_id: UUID, text: str) -> None:
        if not text:
            return
        last = self._buf[-1] if self._buf else None
        if last is not None and last[0] == type and last[1]["entry_id"] == str(entry_id):
            last[1]["text"] += text
        else:
            self._buf.append((type, {"entry_id": str(entry_id), "text": text}))
        self._buf_len += len(text)
        if self._buf_len >= self._chars:
            await self.flush()

    async def tool_delta(
        self, entry_id: UUID, index: int, tool_call_id: str | None, name: str | None, args: str
    ) -> None:
        """Fragment d'arguments d'outil (D105), fusionné avec le précédent du même appel."""
        if not args:
            return
        last = self._buf[-1] if self._buf else None
        if (
            last is not None
            and last[0] == "tool_call_delta"
            and last[1]["entry_id"] == str(entry_id)
            and last[1]["index"] == index
        ):
            last[1]["args_delta"] += args
        else:
            self._buf.append(
                (
                    "tool_call_delta",
                    {
                        "entry_id": str(entry_id),
                        "index": index,
                        "tool_call_id": tool_call_id,
                        "name": name,
                        "args_delta": args,
                    },
                )
            )
        self._buf_len += len(args)
        if self._buf_len >= self._chars:
            await self.flush()

    async def emit(self, type: str, data: dict[str, Any]) -> None:
        async with self._lock:
            await self._write_locked([(type, data)])

    async def flush(self) -> None:
        async with self._lock:
            await self._write_locked([])

    async def _write_locked(self, extra: list[tuple[str, dict[str, Any]]]) -> None:
        items = self._buf + extra
        self._buf, self._buf_len = [], 0
        if not items:
            return
        async with self._storage.transaction() as tx:
            await tx.append_events([self.new(t, d) for t, d in items])
            tx.notify_events(self.session_id)
        self._hub.wake(self.session_id)
