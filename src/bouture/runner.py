"""Exécution des tours en tâche de fond, détachée de la requête HTTP (D22)."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Protocol
from uuid import UUID

log = logging.getLogger("bouture")


class TurnRunner(Protocol):
    def start(self, turn_id: UUID, fn: Callable[[], Awaitable[None]]) -> None: ...
    async def join(self, timeout: float | None = None) -> None:
        """Attend la fin des tours lancés par ce runner."""
        ...


class InProcessRunner:
    """Lance chaque tour dans une ``asyncio.Task`` du processus courant."""

    def __init__(self) -> None:
        self.tasks: dict[UUID, asyncio.Task[None]] = {}

    def start(self, turn_id: UUID, fn: Callable[[], Awaitable[None]]) -> None:
        async def wrapper() -> None:
            try:
                await fn()
            except Exception:
                log.exception("tour %s : erreur inattendue", turn_id)
            finally:
                self.tasks.pop(turn_id, None)

        self.tasks[turn_id] = asyncio.create_task(wrapper(), name=f"bouture-turn-{turn_id}")

    async def join(self, timeout: float | None = None) -> None:
        tasks = list(self.tasks.values())
        if tasks:
            await asyncio.wait(tasks, timeout=timeout)
