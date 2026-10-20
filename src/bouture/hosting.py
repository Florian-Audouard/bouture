"""Hébergement de bouture dans une boucle à elle, sur un thread dédié (D123-D125).

Pour les hôtes dont la boucle ne convient pas : application synchrone (routes FastAPI ``def``,
SQLAlchemy synchrone), ou boucle ``Selector`` imposée sous Windows (psycopg, certains réglages
d'uvicorn), qui ne sait pas lancer de sous-processus. ``BackgroundAgents`` fait tourner
``Agents`` dans sa propre boucle (``Proactor`` sous Windows) et offre des ponts depuis n'importe
quel thread ou boucle.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import queue
import sys
import threading
from collections.abc import AsyncIterator, Callable, Coroutine, Iterator
from concurrent.futures import Future
from typing import Any, Self, cast

from .agents import Agents
from .views import TurnHandle


def default_loop_factory() -> asyncio.AbstractEventLoop:
    """``Proactor`` sous Windows (sous-processus possibles quelle que soit la politique de
    l'hôte), boucle par défaut ailleurs."""
    if sys.platform == "win32":
        return asyncio.ProactorEventLoop()
    return asyncio.new_event_loop()


class BackgroundAgents[D]:
    """``Agents`` hébergé sur un thread dédié avec sa propre boucle asyncio (D123).

    Enregistrer les définitions avant ``start()``. Ensuite, tout passe par les ponts :

    - ``await bg.run(bg.agents.send_message(...))`` depuis une autre boucle (async) ;
    - ``bg.run_sync(bg.agents.get_session(...))`` depuis un thread (route FastAPI ``def``) ;
    - ``bg.stream(it)`` / ``bg.stream_sync(it)`` pour les itérateurs (``handle.events()``,
      ``subscribe``) ;
    - ``bg.aio`` : vue de ``Agents`` dont chaque méthode passe par le pont async (D124).

    ``shutdown()`` clôt les tours en cours (``server_shutdown``), ferme le stockage et arrête le
    thread.
    """

    def __init__(
        self,
        agents: Agents[D],
        *,
        loop_factory: Callable[[], asyncio.AbstractEventLoop] = default_loop_factory,
        auto_migrate: bool = True,
        name: str = "bouture",
        shutdown_timeout: float = 10.0,
    ) -> None:
        self.agents = agents
        self.loop: asyncio.AbstractEventLoop | None = None
        self._loop_factory = loop_factory
        self._auto_migrate = auto_migrate
        self._name = name
        self._shutdown_timeout = shutdown_timeout
        self._thread: threading.Thread | None = None
        self._stopped = False

    # -- cycle de vie --------------------------------------------------------------------
    def start(self) -> Self:
        """Démarre le thread, sa boucle, puis ``agents.setup()`` ; rend la main une fois prêt.

        Une erreur de ``setup`` est relevée ici et le thread s'arrête.
        """
        if self._thread is not None:
            raise RuntimeError("BackgroundAgents déjà démarré")
        ready = threading.Event()
        failure: list[BaseException] = []

        def main() -> None:
            loop = self._loop_factory()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.agents.setup(auto_migrate=self._auto_migrate))
            except BaseException as e:
                failure.append(e)
                loop.close()
                ready.set()
                return
            self.loop = loop
            ready.set()
            try:
                loop.run_forever()
            finally:
                _drain(loop)

        self._thread = threading.Thread(target=main, name=self._name, daemon=True)
        self._thread.start()
        ready.wait()
        if failure:
            self._thread.join()
            self._stopped = True
            raise failure[0]
        return self

    def shutdown(self, timeout: float | None = None) -> None:
        """Clôt les tours (``server_shutdown``), ferme le stockage, arrête la boucle et le thread.

        Idempotent. Bloquant : depuis une boucle, ``await asyncio.to_thread(bg.shutdown)`` ou
        ``async with``.
        """
        loop, thread = self.loop, self._thread
        if loop is None or thread is None or self._stopped:
            return
        if threading.current_thread() is thread:
            raise RuntimeError("shutdown() ne peut pas être appelé depuis la boucle de bouture")
        self._stopped = True
        grace = self._shutdown_timeout if timeout is None else timeout
        fut = asyncio.run_coroutine_threadsafe(self.agents.shutdown(grace), loop)
        with contextlib.suppress(Exception):
            fut.result(grace + 5)
        loop.call_soon_threadsafe(loop.stop)
        thread.join()

    @property
    def running(self) -> bool:
        return self.loop is not None and not self._stopped

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.shutdown()

    async def __aenter__(self) -> Self:
        return await asyncio.to_thread(self.start)

    async def __aexit__(self, *exc: object) -> None:
        await asyncio.to_thread(self.shutdown)

    # -- ponts ---------------------------------------------------------------------------
    def submit[T](self, coro: Coroutine[Any, Any, T]) -> Future[T]:
        """Planifie ``coro`` sur la boucle de bouture ; renvoie un ``concurrent.futures.Future``."""
        loop = self._require(coro)
        return asyncio.run_coroutine_threadsafe(coro, loop)

    async def run[T](self, coro: Coroutine[Any, Any, T]) -> T:
        """Pont async : exécute ``coro`` sur la boucle de bouture et l'attend depuis la boucle
        appelante. Annuler l'appelant annule l'appel. Depuis la boucle de bouture elle-même
        (dans un outil), simple ``await``."""
        if self._on_own_loop():
            return await coro
        return await asyncio.wrap_future(self.submit(coro))

    def run_sync[T](self, coro: Coroutine[Any, Any, T], timeout: float | None = None) -> T:
        """Pont synchrone : bloque le thread appelant jusqu'au résultat (``TimeoutError`` au-delà
        de ``timeout``, l'appel est alors annulé). Interdit depuis la boucle de bouture."""
        if self._on_own_thread():
            coro.close()
            raise RuntimeError("run_sync() bloquerait la boucle de bouture : utilisez await")
        fut = self.submit(coro)
        try:
            return fut.result(timeout)
        except TimeoutError:
            fut.cancel()
            raise

    async def stream[T](self, it: AsyncIterator[T]) -> AsyncIterator[T]:
        """Pont async pour un itérateur de bouture (``handle.events()``, ``subscribe``).

        Fermer l'itérateur renvoyé (``break``, déconnexion) arrête la lecture côté bouture.
        """
        if self._on_own_loop():
            async for item in it:
                yield item
            return
        caller = asyncio.get_running_loop()
        q: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

        def put(item: tuple[str, Any]) -> None:
            with contextlib.suppress(RuntimeError):  # boucle appelante déjà fermée
                caller.call_soon_threadsafe(q.put_nowait, item)

        fut = self.submit(_pump(it, put))
        try:
            while True:
                kind, value = await q.get()
                if kind == "item":
                    yield cast("T", value)
                elif kind == "error":
                    raise cast("BaseException", value)
                else:
                    return
        finally:
            fut.cancel()

    def stream_sync[T](self, it: AsyncIterator[T]) -> Iterator[T]:
        """Pont synchrone pour un itérateur de bouture ; ``close()`` arrête la lecture."""
        if self._on_own_thread():
            raise RuntimeError("stream_sync() bloquerait la boucle de bouture : utilisez stream()")
        q: queue.Queue[tuple[str, Any]] = queue.Queue()
        fut = self.submit(_pump(it, q.put))
        try:
            while True:
                kind, value = q.get()
                if kind == "item":
                    yield cast("T", value)
                elif kind == "error":
                    raise cast("BaseException", value)
                else:
                    return
        finally:
            fut.cancel()

    @property
    def aio(self) -> Agents[D]:
        """Vue de ``agents`` utilisable depuis n'importe quelle boucle (D124) : méthodes async
        passées par ``run``, itérateurs par ``stream``, ``TurnHandle`` rebranchés ; le reste
        (registre, définitions) est lu directement."""
        return cast("Agents[D]", _AsyncView(self))

    # -- interne -------------------------------------------------------------------------
    def _require(self, coro: Coroutine[Any, Any, Any]) -> asyncio.AbstractEventLoop:
        if self.loop is None or self._stopped:
            coro.close()
            raise RuntimeError("BackgroundAgents non démarré ou arrêté")
        return self.loop

    def _on_own_thread(self) -> bool:
        return self._thread is not None and threading.current_thread() is self._thread

    def _on_own_loop(self) -> bool:
        try:
            return asyncio.get_running_loop() is self.loop
        except RuntimeError:
            return False

    def _rebind(self, value: Any) -> Any:
        if isinstance(value, TurnHandle):
            inner = value
            return TurnHandle(inner.session_id, inner.turn_id, lambda: self.stream(inner.events()))
        return value


class _AsyncView:
    def __init__(self, bg: BackgroundAgents[Any]) -> None:
        self._bg = bg

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._bg.agents, name)
        bg = self._bg
        if inspect.isasyncgenfunction(attr):

            def stream(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
                return bg.stream(attr(*args, **kwargs))

            return stream
        if inspect.iscoroutinefunction(attr):

            async def call(*args: Any, **kwargs: Any) -> Any:
                return bg._rebind(await bg.run(attr(*args, **kwargs)))  # pyright: ignore[reportPrivateUsage]

            return call
        return attr


async def _pump(it: AsyncIterator[Any], put: Callable[[tuple[str, Any]], None]) -> None:
    try:
        async for item in it:
            put(("item", item))
        put(("end", None))
    except Exception as e:
        put(("error", e))
    finally:
        aclose = getattr(it, "aclose", None)
        if aclose is not None:
            with contextlib.suppress(Exception):
                await aclose()


def _drain(loop: asyncio.AbstractEventLoop) -> None:
    """Annule ce qui reste sur la boucle (ponts non consommés) puis la ferme."""
    try:
        tasks = asyncio.all_tasks(loop)
        for t in tasks:
            t.cancel()
        if tasks:
            loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
        loop.run_until_complete(loop.shutdown_asyncgens())
    finally:
        loop.close()
