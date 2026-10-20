"""Mesure du coût et de la latence de ``ClaudeCodeClient`` sur une conversation à outils (D127).

Une conversation de ``--steps`` étapes (appels au modèle) : le modèle appelle l'outil ``compter``
une fois par étape, puis répond. Les étapes sont réparties en tours de ``--per-turn`` étapes
(défaut : un seul tour). Pour chaque étape : latence totale, délai du premier token, tokens
d'entrée, ``cache_read``, ``cache_write`` et sortie. Modes comparés : ``transcript`` (défaut de
bouture, un processus par étape) et ``persistent`` (D128, processus gardé dans le tour).

**Consomme le quota de l'abonnement Claude** : rien n'est lancé sans ``--run``. ``--fake``
exécute le même déroulé avec un faux modèle (aucun appel réel), pour vérifier le script.

    uv run python scripts/bench_claude_code.py                    # affiche le plan, sort
    uv run python scripts/bench_claude_code.py --fake             # déroulé à blanc
    uv run python scripts/bench_claude_code.py --run --model haiku --json bench.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import asdict, dataclass
from typing import Any

from bouture import (
    AgentDefinition,
    Agents,
    FakeModelClient,
    FakeReply,
    MemoryStorage,
    ModelClient,
    ToolContext,
)
from bouture.messages import Usage
from bouture.models.base import (
    ModelEvent,
    ModelRequest,
    ReasoningDelta,
    TextDelta,
    ToolCallDelta,
    UsageReport,
)
from bouture.models.fake import Script

OWNER = "bench"
SYSTEM = (
    "Tu es un agent de mesure. Suis exactement les consignes : un seul appel d'outil par "
    "réponse, sans texte autour, puis une réponse finale d'un mot quand c'est demandé."
)
# Résultat d'outil d'une taille réaliste (~400 tokens) : le transcript grossit à chaque étape.
PAYLOAD = " ".join(f"ligne {i} : valeur mesurée, unité, commentaire bref." for i in range(40))


@dataclass
class Step:
    mode: str
    turn: int
    step: int
    latency: float
    first_token: float | None
    input_tokens: int
    cache_read: int
    cache_write: int
    output_tokens: int


class Meter:
    """Enveloppe un ``ModelClient`` : chronomètre chaque appel et relève son usage."""

    def __init__(self, inner: ModelClient, mode: str) -> None:
        self.inner = inner
        self.mode = mode
        self.turn = 0
        self.steps: list[Step] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        started = time.perf_counter()
        first: float | None = None
        usage = Usage()
        async for ev in self.inner.stream(request):
            if first is None and isinstance(ev, TextDelta | ReasoningDelta | ToolCallDelta):
                first = time.perf_counter() - started
            if isinstance(ev, UsageReport):
                usage = ev.usage
            yield ev
        n = sum(1 for s in self.steps if s.turn == self.turn) + 1
        self.steps.append(
            Step(
                self.mode,
                self.turn,
                n,
                time.perf_counter() - started,
                first,
                usage.input_tokens,
                usage.cache_read,
                usage.cache_write,
                usage.output_tokens,
            )
        )

    async def aclose(self) -> None:
        aclose = getattr(self.inner, "aclose", None)
        if aclose is not None:
            await aclose()


def definition(model: str, steps: int) -> AgentDefinition[None, Any]:
    d = AgentDefinition[None, Any](
        "bench", model=model, system_prompt=SYSTEM, max_steps=steps + 2, tool_timeout=None
    )

    @d.tool
    async def compter(ctx: ToolContext[None, Any], n: int) -> str:
        """Enregistre la mesure numéro ``n`` et renvoie le relevé."""
        return f"mesure {n} enregistrée. {PAYLOAD}"

    return d


def turns(steps: int, per_turn: int) -> list[int]:
    """Étapes de chaque tour : ``per_turn - 1`` appels d'outil puis une réponse finale."""
    out: list[int] = []
    left = steps
    while left > 0:
        out.append(min(per_turn, left))
        left -= out[-1]
    return out


def prompt(start: int, calls: int) -> str:
    if calls == 0:
        return "Réponds seulement « fini »."
    return (
        f"Appelle l'outil compter {calls} fois de suite, un appel par réponse, avec n = "
        f"{start}, {start + 1}, … {start + calls - 1}. Ensuite réponds seulement « fini »."
    )


def fake_client(plan: list[int]) -> FakeModelClient:
    replies: list[Script] = []
    n = 1
    for size in plan:
        for _ in range(size - 1):
            replies.append(FakeReply("", tool_calls=[("compter", {"n": n})]))
            n += 1
        replies.append(FakeReply("fini"))
    return FakeModelClient(replies)


async def bench(
    mode: str, make_client: Callable[[], ModelClient], model: str, plan: list[int]
) -> list[Step]:
    meter = Meter(make_client(), mode)
    agents = Agents[Any](storage=MemoryStorage(), model_client=meter)
    agents.register(definition(model, max(plan)))
    await agents.setup()
    try:
        s = await agents.create_session("bench", OWNER)
        n = 1
        for i, size in enumerate(plan, 1):
            meter.turn = i
            end = await (await agents.send_message(s.id, OWNER, prompt(n, size - 1))).wait()
            n += size - 1
            if end.data["reason"] != "completed":
                print(f"[{mode}] tour {i} : {end.data['reason']}", file=sys.stderr)
    finally:
        await agents.shutdown()
    return meter.steps


def table(steps: list[Step]) -> str:
    head = f"{'mode':<11} {'tour':>4} {'étape':>5} {'latence':>8} {'1er tok':>8} "
    head += f"{'entrée':>8} {'c_read':>8} {'c_write':>8} {'sortie':>7}"
    rows = [head, "-" * len(head)]
    for s in steps:
        ft = f"{s.first_token:8.2f}" if s.first_token is not None else f"{'-':>8}"
        rows.append(
            f"{s.mode:<11} {s.turn:>4} {s.step:>5} {s.latency:8.2f} {ft} "
            f"{s.input_tokens:>8} {s.cache_read:>8} {s.cache_write:>8} {s.output_tokens:>7}"
        )
    rows.append("")
    for mode in dict.fromkeys(s.mode for s in steps):
        ms = [s for s in steps if s.mode == mode]
        lat = sum(s.latency for s in ms)
        total_in = sum(s.input_tokens + s.cache_read + s.cache_write for s in ms)
        hit = sum(s.cache_read for s in ms) / total_in if total_in else 0.0
        rows.append(
            f"{mode:<11} {len(ms)} étapes, {lat:.1f} s, entrée {sum(s.input_tokens for s in ms)}"
            f", cache_read {sum(s.cache_read for s in ms)}"
            f", cache_write {sum(s.cache_write for s in ms)}, part lue en cache {hit:.0%}"
        )
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--run", action="store_true", help="lance les appels réels (quota Claude)")
    p.add_argument("--fake", action="store_true", help="déroulé à blanc, faux modèle")
    p.add_argument("--model", default="haiku", help="alias Claude Code (défaut : haiku)")
    p.add_argument("--steps", type=int, default=15)
    p.add_argument("--per-turn", type=int, default=0, help="étapes par tour (0 : un seul tour)")
    p.add_argument("--mode", choices=["transcript", "persistent", "both"], default="both")
    p.add_argument("--json", help="écrit les mesures dans ce fichier")
    args = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):  # console Windows en cp1252
        sys.stdout.reconfigure(encoding="utf-8")  # pyright: ignore[reportAttributeAccessIssue]

    plan = turns(args.steps, args.per_turn or args.steps)
    modes = ["transcript", "persistent"] if args.mode == "both" else [args.mode]
    print(f"Plan : {args.steps} étapes en {len(plan)} tour(s) {plan}, modes {modes}.")
    if not (args.run or args.fake):
        print("Rien n'est lancé : --run pour les appels réels (quota), --fake pour un essai.")
        return 2

    def factory(mode: str) -> Callable[[], ModelClient]:
        if args.fake:
            return lambda: fake_client(plan)
        from bouture.models.claude_code import ClaudeCodeClient

        return lambda: ClaudeCodeClient(persistent=mode == "persistent")

    model = "fake" if args.fake else f"claude-code/{args.model}"
    steps: list[Step] = []
    for mode in modes:
        steps += asyncio.run(bench(mode, factory(mode), model, plan))
    print(table(steps))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump([asdict(s) for s in steps], f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
