"""Codec d'état : sérialisation, validation et migrations paresseuses (D44–D46)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from pydantic import BaseModel, ValidationError

from .errors import StateValidationError, StateVersionError

Migration = Callable[[dict[str, Any]], dict[str, Any]]


def migrate(
    data: dict[str, Any],
    stored_version: int,
    current_version: int,
    migrations: Mapping[int, Migration],
    what: str,
) -> dict[str, Any]:
    """Applique les migrations ``v → v+1`` jusqu'à ``current_version``. Rien n'est réécrit en base."""
    if stored_version > current_version:
        raise StateVersionError(
            f"{what} : version stockée {stored_version} plus récente que le code ({current_version})"
        )
    v = stored_version
    while v < current_version:
        fn = migrations.get(v)
        if fn is None:
            raise StateVersionError(f"{what} : aucune migration depuis la version {v}")
        data = fn(data)
        v += 1
    return data


class StateCodec:
    def __init__(
        self,
        kind: str,
        state_type: type[BaseModel] | None,
        version: int,
        migrations: Mapping[int, Migration],
    ) -> None:
        self.kind = kind
        self.state_type = state_type
        self.version = version
        self.migrations = migrations

    def load(self, data: Any, stored_version: int) -> BaseModel | None:
        if self.state_type is None or data is None:
            return None
        data = migrate(data, stored_version, self.version, self.migrations, f"état {self.kind}")
        return self.validate(data)

    def validate(self, data: Any) -> BaseModel:
        assert self.state_type is not None
        try:
            if isinstance(data, self.state_type):
                return self.state_type.model_validate(data.model_dump(mode="json"))
            return self.state_type.model_validate(data)
        except ValidationError as e:
            raise StateValidationError(str(e)) from e

    @staticmethod
    def dump(state: BaseModel | None) -> dict[str, Any] | None:
        return None if state is None else state.model_dump(mode="json")
