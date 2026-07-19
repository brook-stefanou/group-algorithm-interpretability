"""The minimal .env loader."""

import os

from group_algorithm_interp.dotenv import load_dotenv


def test_missing_file_is_a_noop(tmp_path):
    assert load_dotenv(tmp_path / "nope.env") == 0


def test_loads_keys_and_skips_comments_and_blanks(tmp_path, monkeypatch):
    monkeypatch.delenv("MY_KEY", raising=False)
    monkeypatch.delenv("QUOTED", raising=False)
    env = tmp_path / ".env"
    env.write_text('# a comment\n\nMY_KEY=abc123\nQUOTED="with spaces"\nnot_a_pair_line\n')
    n = load_dotenv(env)
    assert n == 2
    assert os.environ["MY_KEY"] == "abc123"
    assert os.environ["QUOTED"] == "with spaces"  # surrounding quotes stripped


def test_does_not_override_existing_env(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_KEY", "from-real-env")
    (tmp_path / ".env").write_text("MY_KEY=from-dotenv\n")
    load_dotenv(tmp_path / ".env")
    assert os.environ["MY_KEY"] == "from-real-env"  # real env wins
