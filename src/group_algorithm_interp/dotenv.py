"""Minimal, zero-dependency .env loader.

`uv run` does not load a .env file by default, and the code reads keys from
``os.environ``. This loads ``KEY=VALUE`` lines from a .env into the environment
so the optional API keys are picked up -- WITHOUT overriding variables already
set in the real environment (an explicit ``export`` or a CI value always wins).

Deliberately tiny: no interpolation, no multiline values, no ``export`` prefix.
Blank lines and ``#`` comment lines are skipped. Values are stripped of one pair
of surrounding quotes. Inline ``# comments`` are NOT parsed -- keep .env values
on their own (the shipped .env.example does).
"""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: Path | str = ".env") -> int:
    """Load ``path`` into ``os.environ`` (setdefault semantics). Returns the number
    of variables newly set. A missing file is a no-op (returns 0), so this is safe
    to call unconditionally in the base repo."""
    file = Path(path)
    if not file.is_file():
        return 0
    loaded = 0
    for raw in file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value
            loaded += 1
    return loaded
