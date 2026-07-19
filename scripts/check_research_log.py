"""Validate the public research log and render CI-only index artifacts."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from research_log import ResearchLogError, rendered_indexes

ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "docs" / "research-log.md"


def git_show(revision: str) -> str | None:
    """Return the historical log at *revision*, or None when it did not exist."""
    result = subprocess.run(
        ["git", "show", f"{revision}:docs/research-log.md"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout if result.returncode == 0 else None


def check_prefix(base_revision: str) -> list[str]:
    previous = git_show(base_revision)
    if previous is None:
        return []
    current = LOG_PATH.read_text(encoding="utf-8")
    if current.startswith(previous):
        return []
    return [
        "docs/research-log.md is append-only: its content at "
        f"{base_revision} is not an exact prefix of the current file."
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render-dir",
        type=Path,
        help="directory for rendered CI index artifacts",
    )
    parser.add_argument("--base", help="Git revision whose log must be a prefix")
    parser.add_argument("--check-head", action="store_true", help="compare against HEAD")
    args = parser.parse_args()
    try:
        rendered = rendered_indexes(LOG_PATH)
    except ResearchLogError as error:
        print(f"Research log format error: {error}", file=sys.stderr)
        return 1

    failures: list[str] = []
    if args.render_dir:
        args.render_dir.mkdir(parents=True, exist_ok=True)
        for filename, content in rendered.items():
            (args.render_dir / filename).write_text(content, encoding="utf-8")
    if args.base:
        failures.extend(check_prefix(args.base))
    elif args.check_head:
        failures.extend(check_prefix("HEAD"))
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print("Research log format and append-only check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
