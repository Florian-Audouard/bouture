"""``LiteLLMClient`` : identifiants par appel et estimation (D92, D93), sans réseau."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

from bouture import CallContext, ModelError, ModelRequest, UserMessage
from bouture.messages import TextBlock

litellm = pytest.importorskip("litellm")

from bouture.models.litellm import LiteLLMClient  # noqa: E402


def _request() -> ModelRequest:
    return ModelRequest(
        model="openai/gpt-4o-mini",
        system="Sois bref.",
        messages=[UserMessage(content=[TextBlock(text="Bonjour")])],
        context=CallContext("alice", uuid4(), "note"),
    )


class _Stream:
    def __init__(self) -> None:
        delta = SimpleNamespace(content="Salut", tool_calls=None)
        self.chunks = [
            SimpleNamespace(
                usage=None, choices=[SimpleNamespace(delta=delta, finish_reason="stop")]
            )
        ]

    def __aiter__(self) -> _Stream:
        return self

    async def __anext__(self) -> Any:
        if not self.chunks:
            raise StopAsyncIteration
        return self.chunks.pop(0)

    async def aclose(self) -> None:
        pass


async def test_credentials_are_passed_per_call(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any) -> _Stream:
        seen.update(kwargs)
        return _Stream()

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    async def credentials(request: ModelRequest) -> dict[str, Any]:
        assert request.context is not None
        return {"api_key": f"key-of-{request.context.owner}", "api_base": "http://llm.local"}

    client = LiteLLMClient(api_keys={"openai": "server-key"}, credentials=credentials)
    events = [e async for e in client.stream(_request())]
    assert seen["api_key"] == "key-of-alice" and seen["api_base"] == "http://llm.local"
    assert any(getattr(e, "text", None) == "Salut" for e in events)

    async def broken(request: ModelRequest) -> dict[str, Any]:
        raise RuntimeError("coffre-fort indisponible")

    with pytest.raises(ModelError) as info:
        _ = [e async for e in LiteLLMClient(credentials=broken).stream(_request())]
    assert info.value.code == "credentials_error"


async def test_estimate_counts_input_tokens() -> None:
    est = await LiteLLMClient().estimate(_request())
    assert est.input_tokens > 0
    assert est.cost is None or est.cost >= 0
