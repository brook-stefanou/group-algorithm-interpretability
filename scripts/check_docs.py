"""Check that local links in repository Markdown files resolve."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlparse

REPO = Path(__file__).resolve().parent.parent
LINK = re.compile(r"(?<!!)\[[^]]+\]\(([^)]+)\)")


def markdown_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", "*.md"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    return [REPO / name for name in result.stdout.splitlines() if name]


def broken_links(path: Path) -> list[str]:
    broken: list[str] = []
    for raw in LINK.findall(path.read_text(encoding="utf-8")):
        target = raw.strip().split(maxsplit=1)[0].strip("<>")
        parsed = urlparse(target)
        if not target or target.startswith("#") or parsed.scheme or parsed.netloc:
            continue
        relative = unquote(target.split("#", 1)[0])
        resolved = (path.parent / relative).resolve()
        if not resolved.exists():
            broken.append(target)
    return broken


def main() -> int:
    failures = [
        f"{path.relative_to(REPO)}: {target}"
        for path in markdown_files()
        for target in broken_links(path)
    ]
    if failures:
        print("Broken local Markdown links:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("All local Markdown links resolve.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
