"""Modèle de démonstration sans LLM : quelques commandes textuelles déclenchent les outils.

Permet de lancer l'exemple sans clé d'API. Avec ``LLM_API_KEY`` défini, l'app utilise un vrai LLM.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any

from bouture import FakeModelClient, FakeReply, ModelRequest
from bouture.messages import TextBlock, ToolResultMessage, UserMessage
from bouture.models.base import ModelEvent

Args = dict[str, Any]


def _field(field: str) -> Callable[[str], Args]:
    return lambda v: {field: v}


def _items(name: str) -> Callable[[str], Args]:
    return lambda v: {"list": name, "items": [v]}


def _ask(v: str) -> Args:
    """``question: Elfe ou humaine ? | elfe | humaine`` → question et choix."""
    question, *choices = [p.strip() for p in v.split("|")]
    return {"question": question, "choices": choices or None}


# (préfixe, outil, arguments) : la première ligne dont l'outil existe dans le kind l'emporte.
COMMANDS: list[tuple[str, str, Callable[[str], Args]]] = [
    ("nom:", "set_name", _field("name")),
    ("nom:", "set_fields", _field("name")),
    ("âge:", "set_fields", _field("age")),
    ("genre:", "set_fields", _field("gender")),
    ("apparence:", "set_fields", _field("appearance")),
    ("personnalité:", "set_fields", _field("personality")),
    ("histoire:", "set_backstory", _field("backstory")),
    ("histoire:", "set_fields", _field("backstory")),
    ("description:", "set_description", _field("description")),
    ("lieu:", "add_place", _field("place")),
    ("trait:", "add_trait", _field("trait")),
    ("retire:", "remove_trait", _field("trait")),
    ("manie:", "add_items", _items("quirks")),
    ("aime:", "add_items", _items("likes")),
    ("déteste:", "add_items", _items("dislikes")),
    ("compétence:", "add_items", _items("skills")),
    ("question:", "ask", _ask),
    ("portrait:", "propose_portrait", _field("style")),
]


def help_text(tools: set[str]) -> str:
    cmds = list(dict.fromkeys(f"« {c} … »" for c, tool, _ in COMMANDS if tool in tools))
    if "read_world" in tools:
        cmds.append("« monde »")
    return "Agent de démonstration (sans LLM). Commandes : " + ", ".join(cmds) + "."


def _text(m: UserMessage) -> str:
    return " ".join(b.text for b in m.content if isinstance(b, TextBlock)).strip()


def decide(request: ModelRequest) -> FakeReply:
    tools = {t.name for t in request.tools}
    for m in reversed(request.messages):
        if isinstance(m, ToolResultMessage):
            return FakeReply(f"Résultat de {m.name} : {m.text}", delay=0.01)
        if isinstance(m, UserMessage):
            text = _text(m)
            if text.startswith(("<etat_courant>", "[Note système]")):
                continue
            low = text.lower()
            if low == "monde" and "read_world" in tools:
                return FakeReply("Je consulte le monde.", tool_calls=[("read_world", {})])
            for prefix, tool, args in COMMANDS:
                if low.startswith(prefix) and tool in tools:
                    return FakeReply("", tool_calls=[(tool, args(text[len(prefix) :].strip()))])
            break
    return FakeReply(help_text(tools), delay=0.01)


class DemoModelClient:
    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        async for ev in FakeModelClient([decide(request)]).stream(request):
            yield ev
