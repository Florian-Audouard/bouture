"""``scripts/bench_claude_code.py`` : refuse sans ``--run``, déroulé à blanc avec ``--fake``."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "bench_claude_code.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("bench_claude_code", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # pour les dataclasses du script
    spec.loader.exec_module(mod)
    return mod


def test_refuses_without_run(capsys: pytest.CaptureFixture[str]) -> None:
    assert _load().main([]) == 2
    assert "Rien n'est lancé" in capsys.readouterr().out


def test_fake_run(tmp_path: Path) -> None:
    out = tmp_path / "bench.json"
    assert _load().main(["--fake", "--per-turn", "5", "--json", str(out)]) == 0
    steps = json.loads(out.read_text(encoding="utf-8"))
    assert len(steps) == 30  # 15 étapes, deux modes
    assert {s["turn"] for s in steps} == {1, 2, 3}
    assert all(s["input_tokens"] > 0 for s in steps)
