"""Intégration complète dans une application synchrone : SQLite, ``BackgroundAgents``,
``ClaudeCodeClient`` et un outil qui renvoie une image.

    uv run python -m examples.studio "dessine un carré rouge"

Appelle Claude Code (abonnement Claude, quota consommé) ; ``--demo`` rejoue un modèle factice.
Les données vont dans ``.data/studio/`` à la racine (base SQLite et images).
"""

from __future__ import annotations

import struct
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from bouture import AgentDefinition, Agents, ImageContext, ModelClient, ToolContext
from bouture.hosting import BackgroundAgents
from bouture.messages import ImageBlock, TextBlock
from bouture.storage.sqlite import SQLiteStorage

DATA = Path(__file__).resolve().parents[1] / ".data" / "studio"
COLORS = {"rouge": (220, 40, 40), "vert": (40, 160, 70), "bleu": (40, 90, 220)}


class Studio(BaseModel):
    renders: list[str] = []


@dataclass
class Deps:
    images: Path  # dossier des rendus ; la base ne garde que des références (D45)


def png(rgb: tuple[int, int, int], size: int = 64) -> bytes:
    """PNG uni, sans dépendance (le « moteur de rendu » de l'exemple)."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    row = b"\x00" + bytes(rgb) * size
    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(row * size))
        + chunk(b"IEND", b"")
    )


studio = AgentDefinition[Studio, Deps](
    "studio",
    model="claude-code/sonnet",
    state_type=Studio,
    system_prompt="Tu es l'assistant d'un studio d'images. Utilise render pour dessiner.",
)


@studio.tool(timeout=None)  # un vrai rendu peut durer : pas de délai (D121)
async def render(ctx: ToolContext[Studio, Deps], color: str) -> list[TextBlock | ImageBlock]:
    """Dessine un carré de la couleur demandée (rouge, vert ou bleu) et renvoie l'image."""
    rgb = COLORS.get(color.lower())
    if rgb is None:
        raise ValueError(f"couleur inconnue : {color} (rouge, vert ou bleu)")
    ref = f"render-{len(ctx.state.renders) + 1}.png"
    (ctx.deps.images / ref).write_bytes(png(rgb))
    ctx.state.renders.append(ref)
    return [TextBlock(text=f"Carré {color} rendu : {ref}."), ImageBlock(ref=ref)]


def build(client: ModelClient, data: Path = DATA) -> BackgroundAgents[Deps]:
    images = data / "images"
    images.mkdir(parents=True, exist_ok=True)
    agents = Agents[Deps](
        storage=SQLiteStorage(data / "bouture.db"), deps=Deps(images), model_client=client
    )
    agents.register(studio)

    @agents.image_loader
    async def load(ref: str, ctx: ImageContext[Deps]) -> tuple[bytes, str]:  # pyright: ignore[reportUnusedFunction]
        return (ctx.deps.images / ref).read_bytes(), "image/png"

    return BackgroundAgents(agents)


def run(bg: BackgroundAgents[Deps], prompt: str, owner: str = "moi") -> Studio:
    """Code synchrone de l'application : chaque appel passe par un pont (D124)."""
    a = bg.agents
    s = bg.run_sync(a.create_session("studio", owner))
    h = bg.run_sync(a.send_message(s.id, owner, prompt))
    for ev in bg.stream_sync(h.events()):
        if ev.type == "text_delta":
            print(ev.data["text"], end="", flush=True)
        elif ev.type == "tool_result":
            print(f"\n[{ev.data['content']}]")
    print()
    return bg.run_sync(a.get_state(s.id, owner)).state


def main(argv: list[str]) -> None:
    demo = "--demo" in argv
    prompt = " ".join(a for a in argv if a != "--demo") or "dessine un carré rouge"
    client: ModelClient
    if demo:
        from bouture import FakeModelClient, FakeReply

        client = FakeModelClient(
            [FakeReply("", tool_calls=[("render", {"color": "rouge"})]), FakeReply("Voilà.")]
        )
    else:
        from bouture.models.claude_code import ClaudeCodeClient

        client = ClaudeCodeClient()
    with build(client) as bg:  # start() puis shutdown() : tours clos, base fermée
        print(run(bg, prompt))


if __name__ == "__main__":
    main(sys.argv[1:])
