"""Focused tests for `scripts/enumerate_groups.py`'s two-stage pipeline guards.

`scripts/enumerate_groups.py` is an executable entry point, not part of the
installed package, so it is loaded here by file path rather than imported
normally (mirrors `tests/test_run_batch_command.py`). The module never
imports Sage/GAP at module load, so it can be loaded and exercised without
either installed.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "enumerate_groups.py"


def _load_enumerate_groups() -> ModuleType:
    spec = importlib.util.spec_from_file_location("enumerate_groups", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


enumerate_groups = _load_enumerate_groups()


def test_canonical_output_path_differs_from_source_path() -> None:
    assert enumerate_groups.CANONICAL_OUTPUT_PATH != enumerate_groups.SOURCE_GROUP_PROPERTIES_PATH


def test_output_arg_defaults_to_canonical_path() -> None:
    # Parse with no --output given at all: the default must be the canonical
    # enriched file, not the stage-1 source (the original foot-gun).
    args = enumerate_groups._build_arg_parser().parse_args([])
    assert args.output == Path(enumerate_groups.CANONICAL_OUTPUT_PATH)
    assert args.output != args.source_group_properties


def test_check_output_not_source_allows_distinct_paths(tmp_path: Path) -> None:
    source = tmp_path / "group_properties.jsonl"
    output = tmp_path / "group_properties_full.jsonl"
    # Must not raise.
    enumerate_groups._check_output_not_source(output, source)


def test_check_output_not_source_rejects_identical_resolved_paths(tmp_path: Path) -> None:
    source = tmp_path / "group_properties.jsonl"
    output = tmp_path / "group_properties.jsonl"
    with pytest.raises(SystemExit, match="refusing to run"):
        enumerate_groups._check_output_not_source(output, source)


def test_check_output_not_source_rejects_differently_spelled_same_path(tmp_path: Path) -> None:
    source = tmp_path / "group_properties.jsonl"
    output = tmp_path / "sub" / ".." / "group_properties.jsonl"
    with pytest.raises(SystemExit, match="refusing to run"):
        enumerate_groups._check_output_not_source(output, source)


def test_main_refuses_when_output_equals_source(tmp_path: Path, monkeypatch) -> None:
    """The guard fires before any Sage/GAP import, from main()'s own argv."""
    shared = tmp_path / "group_properties.jsonl"
    shared.write_text('{"order": 21, "index": 1}\n', encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "enumerate_groups.py",
            "--source-group-properties",
            str(shared),
            "--output",
            str(shared),
        ],
    )
    with pytest.raises(SystemExit, match="refusing to run"):
        enumerate_groups.main()


def test_read_source_rows_missing_file_gives_regeneration_message(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.jsonl"
    with pytest.raises(SystemExit, match="Regenerate it with: gap"):
        enumerate_groups._read_source_rows(missing)


def test_read_source_rows_reads_present_file(tmp_path: Path) -> None:
    source = tmp_path / "group_properties.jsonl"
    source.write_text(
        "\n".join(
            [
                "# a historical GAP diagnostic comment line",
                json.dumps({"order": 21, "index": 1, "transitive_degree": None}),
                json.dumps({"order": 24, "index": 3, "transitive_degree": None}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rows = enumerate_groups._read_source_rows(source)
    assert set(rows) == {(21, 1), (24, 3)}
