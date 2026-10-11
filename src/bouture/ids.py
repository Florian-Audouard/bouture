"""UUIDv7 monotones (ordre temporel), générés avant l'écriture en base."""

from __future__ import annotations

import os
import threading
import time
import uuid

_lock = threading.Lock()
_last_ms = 0
_counter = 0


def uuid7() -> uuid.UUID:
    """UUIDv7 (RFC 9562) strictement croissant dans un même processus.

    Les 12 bits ``rand_a`` servent de compteur dans une même milliseconde, ce qui garantit
    que l'ordre des identifiants suit l'ordre de création (utilisé pour trier l'arbre).
    """
    global _last_ms, _counter
    with _lock:
        ms = time.time_ns() // 1_000_000
        if ms <= _last_ms:
            ms = _last_ms
            _counter += 1
            if _counter > 0xFFF:
                ms += 1
                _counter = 0
        else:
            _counter = 0
        _last_ms = ms
        counter = _counter
    rand_b = int.from_bytes(os.urandom(8)) & ((1 << 62) - 1)
    value = (ms & ((1 << 48) - 1)) << 80
    value |= 0x7 << 76
    value |= counter << 64
    value |= 0b10 << 62
    value |= rand_b
    return uuid.UUID(int=value)
