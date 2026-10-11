"""Modèle de démonstration sans LLM : quelques commandes textuelles déclenchent les outils.

Permet de lancer l'exemple sans clé d'API. Avec ``BOUTURE_MODEL`` défini, l'app utilise LiteLLM.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from bouture import FakeModelClient, FakeReply, ModelRequest
from bouture.messages import TextBlock, ToolResultMessage, UserMessage
from bouture.models.base import ModelEvent

COMMANDS = {
    "nom:": ("set_name", "name"),
    "description:": ("set_description", "description"),
    "lieu:": ("add_place", "place"),
    "histoire:": ("set_backstory", "backstory"),
    "trait:": ("add_trait", "trait"),
    "portrait:": ("propose_portrait", "style"),
}

HELP = (
    "Agent de démonstration (sans LLM). Commandes : "
    + ", ".join(f"« {c} … »" for c in COMMANDS)
    + ", « monde » (personnage : lit le monde lié)."
)


def _text(m: UserMessage) -> str:
    return " ".join(b.text for b in m.content if isinstance(b, TextBlock)).strip()


def decide(request: ModelRequest) -> FakeReply:
    tools = {t.name for t in request.tools}
    for m in reversed(request.messages):
        if isinstance(m, ToolResultMessage):
            return FakeReply(f"Résultat de {m.name} : {m.content}", delay=0.01)
        if isinstance(m, UserMessage):
            text = _text(m)
            if text.startswith(("<etat_courant>", "[Note système]")):
                continue
            low = text.lower()
            if low == "monde" and "read_world" in tools:
                return FakeReply("Je consulte le monde.", tool_calls=[("read_world", {})])
            for prefix, (tool, arg) in COMMANDS.items():
                if low.startswith(prefix) and tool in tools:
                    value = text[len(prefix) :].strip()
                    return FakeReply("", tool_calls=[(tool, {arg: value})])
            break
    return FakeReply(HELP, delay=0.01)


class DemoModelClient:
    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        async for ev in FakeModelClient([decide(request)]).stream(request):
            yield ev
