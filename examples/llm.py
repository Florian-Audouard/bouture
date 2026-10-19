"""Choix du modèle des exemples, lu dans l'environnement et dans le fichier ``.env`` à la racine.

- ``LLM_API_KEY`` : clé d'un fournisseur compatible OpenAI (OpenRouter par défaut) ;
  absente → modèle de démonstration sans LLM ;
- ``LLM_MODEL`` : identifiant du modèle chez le fournisseur (ex. ``anthropic/claude-sonnet-4.5``) ;
- ``LLM_BASE_URL`` : défaut ``https://openrouter.ai/api/v1`` ;
- ``LLM_PROVIDER=claude-code`` : passe par Claude Code et l'abonnement Claude (extra
  ``claude-code``, usage personnel) ; ``LLM_MODEL`` est alors un alias (défaut ``sonnet``) et
  ``LLM_API_KEY`` n'est pas lu.

Les variables déjà définies dans l'environnement priment sur le ``.env``.
"""

from __future__ import annotations

import os
from pathlib import Path

from bouture import ModelClient

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
OPENROUTER = "https://openrouter.ai/api/v1"


def load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover
        return
    load_dotenv(ENV_FILE)


def client_from_env() -> tuple[ModelClient, str]:
    """Renvoie le client et le nom de modèle à donner aux définitions."""
    if os.environ.get("LLM_PROVIDER") == "claude-code":
        from bouture.models.claude_code import ClaudeCodeClient

        return ClaudeCodeClient(), f"claude-code/{os.environ.get('LLM_MODEL') or 'sonnet'}"
    key = os.environ.get("LLM_API_KEY")
    if not key:
        from .character_world.demo_model import DemoModelClient

        return DemoModelClient(), "demo"
    model = os.environ.get("LLM_MODEL")
    if not model:
        raise RuntimeError(
            "LLM_API_KEY est défini mais pas LLM_MODEL (voir le README, section Exemple)"
        )
    import litellm  # pyright: ignore[reportMissingImports]

    from bouture.models.litellm import LiteLLMClient

    litellm.suppress_debug_info = True
    base = os.environ.get("LLM_BASE_URL") or OPENROUTER
    # Préfixe « openai/ » : LiteLLM parle le protocole OpenAI à ``api_base``.
    return LiteLLMClient(api_base=base, api_key=key), f"openai/{model}"
