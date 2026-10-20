"""Format de messages neutre (D2) : c'est ce qui est stocké dans l'arbre.

Aucun format de fournisseur n'est persisté ; les adaptateurs (``bouture.models``) convertissent.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, TypeAdapter


class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ImageBlock(BaseModel):
    """Image référencée : ``ref`` est une URL ou un identifiant résolu par le projet."""

    type: Literal["image"] = "image"
    ref: str
    media_type: str = "image/png"


class ReasoningBlock(BaseModel):
    type: Literal["reasoning"] = "reasoning"
    text: str
    signature: str | None = None


class ToolCallBlock(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    args: dict[str, Any]


ContentBlock = Annotated[TextBlock | ImageBlock, Field(discriminator="type")]
AssistantBlock = Annotated[TextBlock | ReasoningBlock | ToolCallBlock, Field(discriminator="type")]


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0
    model: str | None = None

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read=self.cache_read + other.cache_read,
            cache_write=self.cache_write + other.cache_write,
            model=other.model or self.model,
        )


class UserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: list[ContentBlock]


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: list[AssistantBlock]
    interrupted: bool = False
    usage: Usage | None = None

    @property
    def tool_calls(self) -> list[ToolCallBlock]:
        return [b for b in self.content if isinstance(b, ToolCallBlock)]

    @property
    def text(self) -> str:
        return "".join(b.text for b in self.content if isinstance(b, TextBlock))


class ToolResultMessage(BaseModel):
    """``content`` : texte, ou blocs texte / image (D112). Les images restent des références."""

    role: Literal["tool"] = "tool"
    tool_call_id: str
    name: str
    content: str | list[ContentBlock]
    is_error: bool = False

    @property
    def text(self) -> str:
        return self.content if isinstance(self.content, str) else content_text(self.content)

    @property
    def images(self) -> list[ImageBlock]:
        return [] if isinstance(self.content, str) else images_of(self.content)


Message = Annotated[UserMessage | AssistantMessage | ToolResultMessage, Field(discriminator="role")]

_content_adapter: TypeAdapter[list[ContentBlock]] = TypeAdapter(list[ContentBlock])


def normalize_content(content: str | list[Any]) -> list[TextBlock | ImageBlock]:
    """Accepte une chaîne ou une liste de blocs (objets ou dicts)."""
    if isinstance(content, str):
        return [TextBlock(text=content)]
    return _content_adapter.validate_python(
        [b.model_dump() if isinstance(b, BaseModel) else b for b in content]
    )


def content_text(content: Sequence[TextBlock | ImageBlock]) -> str:
    return "".join(b.text for b in content if isinstance(b, TextBlock))


def images_of(content: Sequence[TextBlock | ImageBlock]) -> list[ImageBlock]:
    return [b for b in content if isinstance(b, ImageBlock)]


def message_images(messages: Sequence[Message]) -> list[ImageBlock]:
    """Images d'un historique (messages utilisateur et résultats d'outils), dans l'ordre."""
    out: list[ImageBlock] = []
    for m in messages:
        if isinstance(m, UserMessage):
            out += images_of(m.content)
        elif isinstance(m, ToolResultMessage):
            out += m.images
    return out
