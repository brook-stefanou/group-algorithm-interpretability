"""Focused tests for deterministic CI-only research-log renderings."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.research_log import ResearchLogError, parse_log, rendered_indexes

VALID_LOG = """# Research log

## Jul 13

<!-- learning: panel-pivot -->
- A matched pair is too narrow for the current question.
  - It was still useful for building the probes.

<!-- decision: choose-coverage -->
- Use constrained coverage for the panel.
  - The feature schema is still open.
"""


def test_rendered_indexes_copy_tagged_bullets_verbatim(tmp_path: Path) -> None:
    log = tmp_path / "research-log.md"
    log.write_text(VALID_LOG, encoding="utf-8")
    indexes = rendered_indexes(log)
    assert "<!-- learning" not in indexes["learnings.md"]
    assert "## Jul 13\n\n- A matched pair" in indexes["learnings.md"]
    assert "  - It was still useful for building the probes." in indexes["learnings.md"]
    assert "- Use constrained coverage for the panel." in indexes["decisions.md"]


def test_tags_require_an_immediate_bullet() -> None:
    with pytest.raises(ResearchLogError, match="followed immediately"):
        parse_log("## Jul 13\n\n<!-- learning: empty -->\nA paragraph\n")


def test_tags_require_unique_slugs() -> None:
    text = VALID_LOG.replace("decision: choose-coverage", "decision: panel-pivot")
    with pytest.raises(ResearchLogError, match="duplicate tag slug"):
        parse_log(text)


def test_tags_require_lower_case_hyphenated_slugs() -> None:
    text = VALID_LOG.replace("learning: panel-pivot", "learning: Panel Pivot")
    with pytest.raises(ResearchLogError, match="lower-case-slug"):
        parse_log(text)
