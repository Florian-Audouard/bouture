"""Exceptions publiques de bouture."""

from __future__ import annotations


class BoutureError(Exception):
    """Base de toutes les erreurs de la lib."""


class NotFound(BoutureError):
    """Session, entrée ou publication introuvable (ou appartenant à un autre propriétaire)."""


class SessionBusy(BoutureError):
    """Un tour est déjà en cours sur cette session."""


class InvalidTarget(BoutureError):
    """L'entrée visée ne convient pas à l'opération (pas un checkpoint, mauvais type…)."""


class NotAwaitingInput(BoutureError):
    """``answer()`` appelé alors qu'aucune question n'est en attente."""


class StateValidationError(BoutureError):
    """L'état fourni ne respecte pas le type d'état de la définition."""


class StateVersionError(BoutureError):
    """Version d'état stockée plus récente que celle du code, ou migration manquante."""


class InvalidSettings(BoutureError):
    """Réglage de modèle non surchargeable pour ce kind (D92)."""


class UnsupportedProviderTool(BoutureError):
    """Outil fournisseur demandé par un kind mais non supporté par le client et le modèle (D117)."""


class ResolveError(BoutureError):
    """Un input ne peut pas être résolu (référence morte, kind inattendu…)."""


class UnknownKind(BoutureError):
    """Kind non enregistré."""


class SchemaOutdated(BoutureError):
    """Le schéma Postgres de la lib n'est pas à jour et ``auto_migrate=False``."""


class ToolFatal(BoutureError):
    """Levée par un outil pour arrêter le tour (``turn_end{reason: "error"}``)."""


class ModelError(BoutureError):
    """Erreur du fournisseur LLM, normalisée par l'adaptateur."""

    def __init__(self, message: str, *, transient: bool = False, code: str = "model_error"):
        super().__init__(message)
        self.transient = transient
        self.code = code
