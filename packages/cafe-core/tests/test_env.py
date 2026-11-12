"""cafe.load_env reads the .env syntax people write, as python-dotenv does, and never overrides a
variable that is already set."""

import os

import pytest

from cafe import load_env

ENV_FILE = """\
# LLM keys
export CAFE_TEST_EXPORTED=sk-exported
CAFE_TEST_COMMENTED=sk-or-123   # personal key
CAFE_TEST_EMPTY=
CAFE_TEST_DOUBLE="abc'"
CAFE_TEST_SINGLE='https://ollama.com'
CAFE_TEST_HASH_IN_QUOTES="sk-#123"
CAFE_TEST_QUOTED_COMMENTED="quoted"  # note
CAFE_TEST_HASH_UNQUOTED=sk#123
  CAFE_TEST_SPACES = spaced value
not an assignment
"""

EXPECTED = {
    "CAFE_TEST_EXPORTED": "sk-exported",
    "CAFE_TEST_COMMENTED": "sk-or-123",
    "CAFE_TEST_EMPTY": "",
    "CAFE_TEST_DOUBLE": "abc'",
    "CAFE_TEST_SINGLE": "https://ollama.com",
    "CAFE_TEST_HASH_IN_QUOTES": "sk-#123",
    "CAFE_TEST_QUOTED_COMMENTED": "quoted",
    "CAFE_TEST_HASH_UNQUOTED": "sk#123",
    "CAFE_TEST_SPACES": "spaced value",
}


@pytest.fixture
def clean_env(monkeypatch):
    """Unset the test variables now, and remove whatever load_env sets after the test."""
    def clean(*names):
        for name in names:
            monkeypatch.setenv(name, "")  # remembers the original state for the teardown
            monkeypatch.delenv(name)
    return clean


def test_reads_export_comments_and_quotes(tmp_path, clean_env):
    clean_env(*EXPECTED, "export CAFE_TEST_EXPORTED")
    (tmp_path / ".env").write_text(ENV_FILE, encoding="utf-8")
    assert load_env(str(tmp_path)) == str(tmp_path / ".env")
    assert {k: os.environ.get(k) for k in EXPECTED} == EXPECTED
    assert "export CAFE_TEST_EXPORTED" not in os.environ


def test_matches_python_dotenv(tmp_path, clean_env):
    dotenv = pytest.importorskip("dotenv")
    clean_env(*EXPECTED, "export CAFE_TEST_EXPORTED")
    (tmp_path / ".env").write_text(ENV_FILE, encoding="utf-8")
    load_env(str(tmp_path))
    parsed = dotenv.dotenv_values(tmp_path / ".env")
    assert {k: os.environ.get(k) for k in parsed} == parsed


def test_ignores_a_byte_order_mark(tmp_path, clean_env):
    clean_env("CAFE_TEST_BOM", "﻿CAFE_TEST_BOM")
    (tmp_path / ".env").write_text("CAFE_TEST_BOM=bom-key\n", encoding="utf-8-sig")
    load_env(str(tmp_path))
    assert os.environ.get("CAFE_TEST_BOM") == "bom-key"
    assert "﻿CAFE_TEST_BOM" not in os.environ


def test_skips_a_line_without_a_key(tmp_path, clean_env):
    clean_env("CAFE_TEST_AFTER")
    (tmp_path / ".env").write_text("=no-key\nCAFE_TEST_AFTER=1\n", encoding="utf-8")
    load_env(str(tmp_path))  # used to raise OSError: a variable with an empty name
    assert os.environ["CAFE_TEST_AFTER"] == "1"


def test_never_overrides_a_set_variable(tmp_path, clean_env, monkeypatch):
    clean_env("CAFE_TEST_EXPORTED")
    monkeypatch.setenv("CAFE_TEST_COMMENTED", "from-the-shell")
    (tmp_path / ".env").write_text(ENV_FILE, encoding="utf-8")
    load_env(str(tmp_path))
    assert os.environ["CAFE_TEST_COMMENTED"] == "from-the-shell"
    assert os.environ["CAFE_TEST_EXPORTED"] == "sk-exported"


def test_searches_upward_from_start(tmp_path, clean_env):
    clean_env("CAFE_TEST_EXPORTED")
    (tmp_path / ".env").write_text("export CAFE_TEST_EXPORTED=up\n", encoding="utf-8")
    sub = tmp_path / "a" / "b"
    sub.mkdir(parents=True)
    assert load_env(str(sub)) == str(tmp_path / ".env")
    assert os.environ["CAFE_TEST_EXPORTED"] == "up"
