"""Tiny zero-dependency .env loader.

Looks for the nearest ``.env`` file (current directory upward) and loads its
``KEY=VALUE`` lines into ``os.environ`` without overwriting anything already set.
Kept dependency-free on purpose — the engine never requires python-dotenv.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# A value in matching quotes, optionally followed by a comment: "sk-#1"  # note -> sk-#1
_QUOTED = re.compile(r"""(["'])(.*?)\1\s*(?:#.*)?""")


def _parse_line(raw: str) -> tuple[str, str] | None:
    """``(key, value)`` for one ``.env`` line, or ``None`` when the line sets nothing."""
    line = raw.strip()
    export = re.match(r"export\s+", line)
    if export:
        line = line[export.end():]
    if not line or line.startswith("#") or "=" not in line:
        return None
    key, value = (part.strip() for part in line.split("=", 1))
    if not key:
        return None
    quoted = _QUOTED.fullmatch(value)
    if quoted:
        return key, quoted.group(2)  # one pair of quotes off; a '#' inside them stays
    return key, re.sub(r"\s+#.*", "", value)  # an unquoted value ends at a " #" comment


def load_env(start: str | None = None) -> str | None:
    """Load the nearest ``.env`` file into ``os.environ``.

    Looks in ``start`` and then in each parent directory. Existing environment variables
    win (we only ``setdefault``), so real env vars and CI secrets always override a local
    ``.env``. The usual shell / python-dotenv syntax is understood: blank lines and ``#``
    comment lines are skipped, a leading ``export`` is ignored, one pair of matching single
    or double quotes around a value is removed (a ``#`` inside them is part of the value),
    an unquoted value ends where a ``#`` comment starts after whitespace, and a UTF-8
    byte-order mark is ignored.

    Parameters
    ----------
    start : str, optional
        The directory to start looking in; the current working directory by default.

    Returns
    -------
    str or None
        The path of the ``.env`` file that was loaded, or ``None`` if there is none.
    """
    base = Path(start or os.getcwd()).resolve()
    for cur in [base, *base.parents]:
        candidate = cur / ".env"
        if candidate.is_file():
            for raw in candidate.read_text(encoding="utf-8-sig").splitlines():
                parsed = _parse_line(raw)
                if parsed is not None:
                    os.environ.setdefault(*parsed)
            return str(candidate)
    return None
