"""Parse and deterministically render tagged entries in the public research log."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

DATE_HEADING = re.compile(r"^## ([A-Z][a-z]{2} \d{1,2}(?:, \d{4})?)$")
TAG = re.compile(r"^<!-- (decision|learning): ([a-z0-9-]+) -->$")
TAG_PREFIX = re.compile(r"^<!-- (?:decision|learning):")
TOP_LEVEL_BULLET = re.compile(r"^- \S")
CONTINUATION = re.compile(r"^(?: {2,}|\t|$)")


class ResearchLogError(ValueError):
    """Raised when the public log does not meet its small tag grammar."""


@dataclass(frozen=True)
class LogBlock:
    """One tagged, exact-copy block from a dated log entry."""

    date: str
    kind: str
    slug: str
    lines: tuple[str, ...]


def parse_log(text: str) -> list[LogBlock]:
    """Validate *text* and return its optional decision and learning blocks."""
    lines = text.splitlines()
    blocks: list[LogBlock] = []
    slugs: set[str] = set()
    current_date: str | None = None
    index = 0

    while index < len(lines):
        date_match = DATE_HEADING.match(lines[index])
        if date_match:
            current_date = date_match.group(1)
            index += 1
            continue

        tag_match = TAG.match(lines[index])
        if tag_match is None:
            if TAG_PREFIX.match(lines[index]):
                raise ResearchLogError(
                    f"line {index + 1}: tags must use '<!-- decision: lower-case-slug -->' "
                    "or '<!-- learning: lower-case-slug -->'"
                )
            index += 1
            continue
        if current_date is None:
            raise ResearchLogError(f"line {index + 1}: a tag needs a dated ## heading")

        kind, slug = tag_match.groups()
        if slug in slugs:
            raise ResearchLogError(f"line {index + 1}: duplicate tag slug '{slug}'")
        slugs.add(slug)
        tag_line = index + 1
        index += 1
        if index == len(lines) or not TOP_LEVEL_BULLET.match(lines[index]):
            raise ResearchLogError(
                f"line {tag_line}: tag '{slug}' must be followed immediately by a Markdown bullet"
            )

        block_lines: list[str] = []
        while index < len(lines):
            if DATE_HEADING.match(lines[index]) or TAG.match(lines[index]):
                break
            if TAG_PREFIX.match(lines[index]):
                raise ResearchLogError(
                    f"line {index + 1}: tags must use a lower-case, hyphenated slug"
                )
            if not (TOP_LEVEL_BULLET.match(lines[index]) or CONTINUATION.match(lines[index])):
                raise ResearchLogError(
                    f"line {index + 1}: tagged block '{slug}' contains non-bullet text"
                )
            block_lines.append(lines[index])
            index += 1

        while block_lines and not block_lines[-1]:
            block_lines.pop()
        blocks.append(LogBlock(current_date, kind, slug, tuple(block_lines)))

    return blocks


def render_index(blocks: list[LogBlock], kind: str) -> str:
    """Render an exact-copy index for one tag kind."""
    heading = "Decisions" if kind == "decision" else "Learnings"
    selected = [block for block in blocks if block.kind == kind]
    output = [
        f"# {heading}",
        "",
        "This CI artifact is rendered from tagged entries in "
        "[`research-log.md`](research-log.md). It copies the dated bullet blocks "
        "exactly; edit the research log rather than this artifact.",
        "",
    ]
    if not selected:
        return "\n".join([*output, "No entries yet.", ""])

    for block in selected:
        output.extend([f"## {block.date}", "", *block.lines, ""])
    return "\n".join(output)


def rendered_indexes(log_path: Path) -> dict[str, str]:
    """Return CI artifact filenames and contents for a research-log path."""
    blocks = parse_log(log_path.read_text(encoding="utf-8"))
    return {
        "decisions.md": render_index(blocks, "decision"),
        "learnings.md": render_index(blocks, "learning"),
    }
