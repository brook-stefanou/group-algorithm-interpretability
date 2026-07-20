"""Focused tests for `scripts/run_campaign.py`.

Offline by construction: no subprocess spawns real training. The `run`
callable `run_campaign` accepts is a recording double (mirroring how
`tests/test_sync_runs.py` exercises `sync_rclone`'s injectable `run`), and
`main()` end-to-end tests monkeypatch the module's `subprocess.run`.
`scripts/run_batch.py` is a concurrent build; nothing here imports it or
requires it to exist -- these tests pin `run_campaign.py`'s side of the
contract (the composed overrides and `--seeds` range it passes, and how it
reads exit codes), not `run_batch`'s internals.

The script is an executable entry point, not part of the installed package,
so it is loaded by file path (mirrors `tests/test_sync_runs.py` and
`tests/test_run_batch_command.py`).

The group-properties ground truth (`data/group_properties_full.jsonl`) is
gitignored and absent in CI, so validation tests use a small fixture
catalogue; the one test against the real file skips when it is not present.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from group_algorithm_interp.config import (
    DataConfig,
    GroupSpec,
    LoggingConfig,
    ModelConfig,
    ProjectConfig,
)
from group_algorithm_interp.manifest import (
    create_manifest,
    create_run_dir,
    finalize_manifest,
    save_resolved_config,
)

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "run_campaign.py"
_REAL_CAMPAIGN = Path(__file__).resolve().parent.parent / "configs" / "campaign" / "core.yaml"
_REAL_GROUP_PROPERTIES = (
    Path(__file__).resolve().parent.parent / "data" / "group_properties_full.jsonl"
)


def _load_run_campaign() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_campaign", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_campaign_module = _load_run_campaign()


# --- fixtures ----------------------------------------------------------------


def _cell_dict(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "phase": "pilot",
        "order": 32,
        "index": 18,
        "name": "D32",
        "width": 64,
        "seeds": "0:2",
        "experiment": "core",
        "note": "a fixture cell",
    }
    base.update(overrides)
    return base


def _write_campaign(path: Path, cells: list[dict[str, object]]) -> Path:
    path.write_text(yaml.safe_dump({"cells": cells}))
    return path


def _write_group_properties(path: Path, groups: list[tuple[int, int]]) -> Path:
    lines = [
        json.dumps({"order": order, "index": index, "name": f"G({order},{index})"})
        for order, index in groups
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def _completed_run(
    runs_dir: Path,
    run_id: str,
    *,
    order: int,
    index: int,
    width: int,
    seed: int,
    status: str = "completed",
) -> Path:
    """A run directory carrying the real manifest + resolved-config shapes the
    completion scan reads, for the given cell coordinates."""
    config = ProjectConfig(
        seed=seed,
        model=ModelConfig(d_model=width),
        data=DataConfig(group=GroupSpec(order=order, index=index)),
        logging=LoggingConfig(mode="disabled"),
    )
    run_dir = create_run_dir(run_id, root=runs_dir)
    create_manifest(config, run_dir)
    if status != "running":
        finalize_manifest(run_dir, status=status)
    save_resolved_config(config, run_dir)
    return run_dir


class _RecordingRun:
    """A subprocess.run double: records every command, returns scripted exit
    codes (default 0)."""

    def __init__(self, returncodes: dict[int, int] | None = None) -> None:
        self.commands: list[list[str]] = []
        self.returncodes = returncodes or {}

    def __call__(self, command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        position = len(self.commands)
        self.commands.append(list(command))
        return subprocess.CompletedProcess(command, self.returncodes.get(position, 0))


# --- seed-range parsing --------------------------------------------------------


def test_parse_seed_range_start_stop_exclusive():
    assert list(run_campaign_module.parse_seed_range("0:50")) == list(range(50))
    assert list(run_campaign_module.parse_seed_range("10:12")) == [10, 11]


@pytest.mark.parametrize("bad", ["", "5", "0:0", "5:3", "a:b", "0:10:2", "0-10"])
def test_parse_seed_range_rejects_malformed_specs(bad):
    with pytest.raises(ValueError):
        run_campaign_module.parse_seed_range(bad)


# --- campaign-file parsing ------------------------------------------------------


def test_parse_campaign_reads_cells_in_file_order(tmp_path):
    path = _write_campaign(
        tmp_path / "campaign.yaml",
        [
            _cell_dict(order=32, index=18, name="D32"),
            _cell_dict(phase="core", order=27, index=3, name="(C3 x C3) : C3", seeds="0:50"),
        ],
    )
    cells = run_campaign_module.parse_campaign(path)
    assert [(cell.order, cell.index) for cell in cells] == [(32, 18), (27, 3)]
    assert cells[0].phase == "pilot" and cells[1].phase == "core"
    assert cells[1].seed_range == range(0, 50)
    assert cells[0].experiment == "core"


def test_parse_campaign_rejects_missing_fields(tmp_path):
    cell = _cell_dict()
    del cell["width"]
    path = _write_campaign(tmp_path / "campaign.yaml", [cell])
    with pytest.raises(ValueError, match="width"):
        run_campaign_module.parse_campaign(path)


def test_parse_campaign_rejects_unknown_phase(tmp_path):
    path = _write_campaign(tmp_path / "campaign.yaml", [_cell_dict(phase="warmup")])
    with pytest.raises(ValueError, match="phase"):
        run_campaign_module.parse_campaign(path)


def test_parse_campaign_rejects_bad_seed_spec(tmp_path):
    path = _write_campaign(tmp_path / "campaign.yaml", [_cell_dict(seeds="50:0")])
    with pytest.raises(ValueError, match="seeds"):
        run_campaign_module.parse_campaign(path)


def test_parse_campaign_rejects_non_list_cells(tmp_path):
    path = tmp_path / "campaign.yaml"
    path.write_text("cells: not-a-list\n")
    with pytest.raises(ValueError, match="cells"):
        run_campaign_module.parse_campaign(path)


# --- pre-registration validation -------------------------------------------------


def test_validate_cells_accepts_catalogued_groups_and_names_unknown_ones(tmp_path):
    catalogue = run_campaign_module.load_known_groups(
        _write_group_properties(tmp_path / "groups.jsonl", [(32, 18), (27, 3)])
    )
    good = run_campaign_module.parse_campaign(
        _write_campaign(tmp_path / "good.yaml", [_cell_dict(order=32, index=18)])
    )
    assert run_campaign_module.validate_cells(good, catalogue) == []

    bad = run_campaign_module.parse_campaign(
        _write_campaign(
            tmp_path / "bad.yaml",
            [_cell_dict(order=32, index=18), _cell_dict(order=99, index=1, name="ghost")],
        )
    )
    errors = run_campaign_module.validate_cells(bad, catalogue)
    assert len(errors) == 1
    assert "SmallGroup(99,1)" in errors[0]


def test_main_exits_1_when_a_cell_references_an_unknown_group(tmp_path, capsys):
    campaign = _write_campaign(tmp_path / "campaign.yaml", [_cell_dict(order=99, index=1)])
    groups = _write_group_properties(tmp_path / "groups.jsonl", [(32, 18)])
    code = run_campaign_module.main(
        [
            "--campaign",
            str(campaign),
            "--group-properties",
            str(groups),
            "--runs-dir",
            str(tmp_path / "runs"),
            "--dry-run",
        ]
    )
    assert code == 1
    assert "SmallGroup(99,1)" in capsys.readouterr().err


def test_main_warns_but_continues_when_group_properties_file_is_absent(tmp_path, capsys):
    campaign = _write_campaign(tmp_path / "campaign.yaml", [_cell_dict()])
    code = run_campaign_module.main(
        [
            "--campaign",
            str(campaign),
            "--group-properties",
            str(tmp_path / "missing.jsonl"),
            "--runs-dir",
            str(tmp_path / "runs"),
            "--dry-run",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "skipping the group-catalogue check" in captured.err
    assert "WOULD RUN" in captured.out


# --- skip-complete detection -----------------------------------------------------


def test_cell_is_complete_only_when_every_seed_has_a_completed_manifest(tmp_path):
    runs_dir = tmp_path / "runs"
    _completed_run(runs_dir, "run-a", order=32, index=18, width=64, seed=0)
    _completed_run(runs_dir, "run-b", order=32, index=18, width=64, seed=1)
    index = run_campaign_module.build_completion_index(runs_dir)

    both_done = run_campaign_module.Cell(
        phase="pilot", order=32, index=18, name="D32", width=64, seeds="0:2", experiment="core"
    )
    needs_seed_2 = run_campaign_module.Cell(
        phase="pilot", order=32, index=18, name="D32", width=64, seeds="0:3", experiment="core"
    )
    assert run_campaign_module.cell_is_complete(both_done, index)
    assert not run_campaign_module.cell_is_complete(needs_seed_2, index)


def test_completion_scan_ignores_non_completed_and_mismatched_runs(tmp_path):
    runs_dir = tmp_path / "runs"
    # Right cell, but failed / still running: never counts.
    _completed_run(runs_dir, "run-failed", order=32, index=18, width=64, seed=0, status="failed")
    _completed_run(runs_dir, "run-running", order=32, index=18, width=64, seed=1, status="running")
    # Completed, but a different width / group: a different cell entirely.
    _completed_run(runs_dir, "run-other-width", order=32, index=18, width=128, seed=0)
    _completed_run(runs_dir, "run-other-group", order=32, index=19, width=64, seed=0)
    index = run_campaign_module.build_completion_index(runs_dir)

    cell = run_campaign_module.Cell(
        phase="pilot", order=32, index=18, name="D32", width=64, seeds="0:2", experiment="core"
    )
    assert not run_campaign_module.cell_is_complete(cell, index)
    # The other-width/other-group completions were indexed under their own keys.
    assert index[(32, 18, 128)] == {0}
    assert index[(32, 19, 64)] == {0}


def test_completion_scan_of_missing_runs_dir_is_empty(tmp_path):
    assert run_campaign_module.build_completion_index(tmp_path / "nope") == {}


def test_complete_cells_are_skipped_and_not_invoked(tmp_path, capsys):
    runs_dir = tmp_path / "runs"
    for seed in range(2):
        _completed_run(runs_dir, f"done-{seed}", order=32, index=18, width=64, seed=seed)
    cells = run_campaign_module.parse_campaign(
        _write_campaign(
            tmp_path / "campaign.yaml",
            [
                _cell_dict(order=32, index=18, width=64, seeds="0:2"),  # complete
                _cell_dict(order=32, index=20, name="Q32", width=64, seeds="0:2"),  # not
            ],
        )
    )
    recorder = _RecordingRun()
    outcomes = run_campaign_module.run_campaign(
        cells,
        completion_index=run_campaign_module.build_completion_index(runs_dir),
        run_batch_path=Path("scripts/run_batch.py"),
        extra_overrides=[],
        keep_going=False,
        dry_run=False,
        cwd=tmp_path,
        run=recorder,
    )
    assert [outcome.status for outcome in outcomes] == ["skip", "complete"]
    assert len(recorder.commands) == 1  # only the incomplete cell ran
    assert "data.group.index=20" in recorder.commands[0]


# --- invocation ordering + override construction ----------------------------------


def test_cells_are_invoked_in_campaign_file_order_with_composed_overrides(tmp_path):
    cells = run_campaign_module.parse_campaign(
        _write_campaign(
            tmp_path / "campaign.yaml",
            [
                _cell_dict(order=32, index=18, width=64, seeds="0:10"),
                _cell_dict(order=32, index=18, width=128, seeds="0:10"),
                _cell_dict(phase="core", order=27, index=3, width=128, seeds="0:50"),
            ],
        )
    )
    recorder = _RecordingRun()
    run_campaign_module.run_campaign(
        cells,
        completion_index={},
        run_batch_path=Path("scripts/run_batch.py"),
        extra_overrides=["logging.mode=online"],
        keep_going=False,
        dry_run=False,
        cwd=tmp_path,
        run=recorder,
    )
    assert len(recorder.commands) == 3
    first = recorder.commands[0]
    # The run_batch.py contract: Hydra overrides exactly as scripts/run.py
    # takes them, then --seeds START:STOP.
    assert first[:4] == ["uv", "run", "python", "scripts/run_batch.py"]
    assert "experiment=core" in first
    assert "data.group.order=32" in first
    assert "data.group.index=18" in first
    assert "model.d_model=64" in first
    assert "logging.mode=online" in first
    assert first[-2:] == ["--seeds", "0:10"]
    # Hydra last-wins: the operator override comes after the cell's own tokens.
    assert first.index("logging.mode=online") > first.index("model.d_model=64")
    assert "model.d_model=128" in recorder.commands[1]
    assert "data.group.order=27" in recorder.commands[2]
    assert recorder.commands[2][-2:] == ["--seeds", "0:50"]


# --- --dry-run ---------------------------------------------------------------------


def test_dry_run_prints_per_cell_status_and_runs_nothing(tmp_path, capsys, monkeypatch):
    runs_dir = tmp_path / "runs"
    for seed in range(2):
        _completed_run(runs_dir, f"done-{seed}", order=32, index=18, width=64, seed=seed)
    campaign = _write_campaign(
        tmp_path / "campaign.yaml",
        [
            _cell_dict(order=32, index=18, width=64, seeds="0:2"),
            _cell_dict(order=32, index=20, name="Q32", width=64, seeds="0:2"),
        ],
    )
    groups = _write_group_properties(tmp_path / "groups.jsonl", [(32, 18), (32, 20)])

    def _forbidden(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise AssertionError("--dry-run must not spawn any subprocess")

    monkeypatch.setattr(run_campaign_module.subprocess, "run", _forbidden)
    code = run_campaign_module.main(
        [
            "--campaign",
            str(campaign),
            "--group-properties",
            str(groups),
            "--runs-dir",
            str(runs_dir),
            "--dry-run",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "SKIP (already complete)" in out
    assert "WOULD RUN" in out
    assert "campaign summary: skip=1, would_run=1" in out


def test_only_phase_filters_cells(tmp_path, capsys):
    campaign = _write_campaign(
        tmp_path / "campaign.yaml",
        [
            _cell_dict(order=32, index=18),
            _cell_dict(phase="core", order=27, index=3, seeds="0:50"),
        ],
    )
    groups = _write_group_properties(tmp_path / "groups.jsonl", [(32, 18), (27, 3)])
    code = run_campaign_module.main(
        [
            "--campaign",
            str(campaign),
            "--group-properties",
            str(groups),
            "--runs-dir",
            str(tmp_path / "runs"),
            "--only-phase",
            "core",
            "--dry-run",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "SmallGroup(27,3)" in out
    assert "SmallGroup(32,18)" not in out


def test_only_phase_with_no_matching_cells_is_an_error(tmp_path, capsys):
    campaign = _write_campaign(tmp_path / "campaign.yaml", [_cell_dict(phase="pilot")])
    groups = _write_group_properties(tmp_path / "groups.jsonl", [(32, 18)])
    code = run_campaign_module.main(
        [
            "--campaign",
            str(campaign),
            "--group-properties",
            str(groups),
            "--runs-dir",
            str(tmp_path / "runs"),
            "--only-phase",
            "core",
            "--dry-run",
        ]
    )
    assert code == 1
    assert "no cells in phase" in capsys.readouterr().err


# --- --keep-going + exit codes -----------------------------------------------------


def _three_cells(tmp_path: Path) -> list:
    return run_campaign_module.parse_campaign(
        _write_campaign(
            tmp_path / "campaign.yaml",
            [
                _cell_dict(order=32, index=18, width=64),
                _cell_dict(order=32, index=19, name="QD32", width=64),
                _cell_dict(order=32, index=20, name="Q32", width=64),
            ],
        )
    )


def test_without_keep_going_a_failure_stops_new_cells_but_accounts_for_all(tmp_path):
    recorder = _RecordingRun(returncodes={0: 3})  # first cell fails
    outcomes = run_campaign_module.run_campaign(
        _three_cells(tmp_path),
        completion_index={},
        run_batch_path=Path("scripts/run_batch.py"),
        extra_overrides=[],
        keep_going=False,
        dry_run=False,
        cwd=tmp_path,
        run=recorder,
    )
    assert [outcome.status for outcome in outcomes] == ["failed", "not_run", "not_run"]
    assert len(recorder.commands) == 1
    assert outcomes[0].returncode == 3


def test_with_keep_going_a_failure_does_not_stop_the_remaining_cells(tmp_path):
    recorder = _RecordingRun(returncodes={0: 1})
    outcomes = run_campaign_module.run_campaign(
        _three_cells(tmp_path),
        completion_index={},
        run_batch_path=Path("scripts/run_batch.py"),
        extra_overrides=[],
        keep_going=True,
        dry_run=False,
        cwd=tmp_path,
        run=recorder,
    )
    assert [outcome.status for outcome in outcomes] == ["failed", "complete", "complete"]
    assert len(recorder.commands) == 3


def test_main_exits_nonzero_when_any_cell_failed_even_with_keep_going(tmp_path, monkeypatch):
    campaign = _write_campaign(
        tmp_path / "campaign.yaml",
        [_cell_dict(order=32, index=18), _cell_dict(order=32, index=20, name="Q32")],
    )
    groups = _write_group_properties(tmp_path / "groups.jsonl", [(32, 18), (32, 20)])
    calls: list[list[str]] = []

    def _fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(list(command))
        return subprocess.CompletedProcess(command, 1 if len(calls) == 1 else 0)

    monkeypatch.setattr(run_campaign_module.subprocess, "run", _fake_run)
    code = run_campaign_module.main(
        [
            "--campaign",
            str(campaign),
            "--group-properties",
            str(groups),
            "--runs-dir",
            str(tmp_path / "runs"),
            "--keep-going",
        ]
    )
    assert code == 1
    assert len(calls) == 2  # --keep-going ran the second cell anyway


def test_main_exits_zero_when_every_cell_completes(tmp_path, monkeypatch):
    campaign = _write_campaign(tmp_path / "campaign.yaml", [_cell_dict(order=32, index=18)])
    groups = _write_group_properties(tmp_path / "groups.jsonl", [(32, 18)])
    monkeypatch.setattr(
        run_campaign_module.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0),
    )
    code = run_campaign_module.main(
        [
            "--campaign",
            str(campaign),
            "--group-properties",
            str(groups),
            "--runs-dir",
            str(tmp_path / "runs"),
        ]
    )
    assert code == 0


def test_main_exits_zero_when_every_cell_is_already_complete(tmp_path, monkeypatch, capsys):
    runs_dir = tmp_path / "runs"
    for seed in range(2):
        _completed_run(runs_dir, f"done-{seed}", order=32, index=18, width=64, seed=seed)
    campaign = _write_campaign(
        tmp_path / "campaign.yaml", [_cell_dict(order=32, index=18, width=64, seeds="0:2")]
    )
    groups = _write_group_properties(tmp_path / "groups.jsonl", [(32, 18)])

    def _forbidden(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise AssertionError("a fully complete campaign must not spawn anything")

    monkeypatch.setattr(run_campaign_module.subprocess, "run", _forbidden)
    code = run_campaign_module.main(
        [
            "--campaign",
            str(campaign),
            "--group-properties",
            str(groups),
            "--runs-dir",
            str(runs_dir),
        ]
    )
    assert code == 0
    assert "campaign summary: skip=1" in capsys.readouterr().out


# --- --include-bonus ----------------------------------------------------------------


def _campaign_with_bonus(tmp_path: Path) -> tuple[Path, Path]:
    campaign = _write_campaign(
        tmp_path / "campaign.yaml",
        [
            _cell_dict(order=32, index=18, width=64),
            _cell_dict(phase="core", order=27, index=3, width=128, seeds="0:50"),
            _cell_dict(phase="bonus-e6", order=64, index=92, name="E6a", width=128, seeds="0:50"),
            _cell_dict(phase="bonus-e2", order=128, index=197, name="E2a", width=128, seeds="0:50"),
        ],
    )
    groups = _write_group_properties(
        tmp_path / "groups.jsonl", [(32, 18), (27, 3), (64, 92), (128, 197)]
    )
    return campaign, groups


def test_bonus_cells_are_excluded_entirely_without_include_bonus(tmp_path, capsys, monkeypatch):
    campaign, groups = _campaign_with_bonus(tmp_path)

    def _forbidden(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise AssertionError("--dry-run must not spawn any subprocess")

    monkeypatch.setattr(run_campaign_module.subprocess, "run", _forbidden)
    code = run_campaign_module.main(
        [
            "--campaign",
            str(campaign),
            "--group-properties",
            str(groups),
            "--runs-dir",
            str(tmp_path / "runs"),
            "--dry-run",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    # Excluded from the listing entirely, not merely not run.
    assert "bonus" not in out
    assert "campaign summary: would_run=2" in out


def test_include_bonus_appends_bonus_cells_after_core(tmp_path, capsys):
    campaign, groups = _campaign_with_bonus(tmp_path)
    code = run_campaign_module.main(
        [
            "--campaign",
            str(campaign),
            "--group-properties",
            str(groups),
            "--runs-dir",
            str(tmp_path / "runs"),
            "--include-bonus",
            "--dry-run",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "campaign summary: would_run=4" in out
    assert out.index("core ") < out.index("bonus-e6") < out.index("bonus-e2")


def test_only_phase_bonus_without_include_bonus_errors_with_a_hint(tmp_path, capsys):
    campaign, groups = _campaign_with_bonus(tmp_path)
    code = run_campaign_module.main(
        [
            "--campaign",
            str(campaign),
            "--group-properties",
            str(groups),
            "--runs-dir",
            str(tmp_path / "runs"),
            "--only-phase",
            "bonus-e6",
            "--dry-run",
        ]
    )
    assert code == 1
    assert "bonus phases require --include-bonus" in capsys.readouterr().err


def test_only_phase_bonus_with_include_bonus_runs_that_phase(tmp_path, capsys):
    campaign, groups = _campaign_with_bonus(tmp_path)
    code = run_campaign_module.main(
        [
            "--campaign",
            str(campaign),
            "--group-properties",
            str(groups),
            "--runs-dir",
            str(tmp_path / "runs"),
            "--include-bonus",
            "--only-phase",
            "bonus-e6",
            "--dry-run",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "campaign summary: would_run=1" in out
    assert "SmallGroup(64,92)" in out


# --- --shard ------------------------------------------------------------------------


def test_parse_shard_accepts_zero_based_index():
    assert run_campaign_module.parse_shard("0/8") == (0, 8)
    assert run_campaign_module.parse_shard("7/8") == (7, 8)
    assert run_campaign_module.parse_shard("0/1") == (0, 1)


@pytest.mark.parametrize("bad", ["", "8", "8/8", "-1/8", "a/b", "1/0", "1/2/3"])
def test_parse_shard_rejects_malformed_specs(bad):
    with pytest.raises(ValueError):
        run_campaign_module.parse_shard(bad)


def test_shards_are_deterministic_disjoint_and_cover_every_cell():
    cells = run_campaign_module.parse_campaign(_REAL_CAMPAIGN)
    n = 8
    shards = [run_campaign_module.shard_cells(cells, i, n) for i in range(n)]
    again = [run_campaign_module.shard_cells(cells, i, n) for i in range(n)]
    assert shards == again  # deterministic
    key = lambda cell: (cell.phase, cell.order, cell.index, cell.width)  # noqa: E731
    flattened = [key(cell) for shard in shards for cell in shard]
    assert len(flattened) == len(set(flattened))  # disjoint
    assert sorted(flattened) == sorted(key(cell) for cell in cells)  # complete cover


def test_shards_are_roughly_cost_balanced():
    cells = run_campaign_module.parse_campaign(_REAL_CAMPAIGN)
    n = 8
    costs = [
        sum(run_campaign_module.estimated_cost(cell) for cell in shard)
        for shard in (run_campaign_module.shard_cells(cells, i, n) for i in range(n))
    ]
    mean = sum(costs) / n
    # Round-robin over descending cost: no shard should be wildly heavy or
    # near-empty. (Measured on the checked-in file: max/mean ~ 1.23.)
    assert max(costs) <= 1.5 * mean
    assert min(costs) >= 0.5 * mean


def test_each_shard_preserves_phase_ordering():
    cells = run_campaign_module.parse_campaign(_REAL_CAMPAIGN)
    for n in (2, 3, 8):
        for i in range(n):
            shard = run_campaign_module.shard_cells(cells, i, n)
            ranks = [run_campaign_module.PHASES.index(cell.phase) for cell in shard]
            assert ranks == sorted(ranks)


def test_single_shard_is_the_whole_list_in_campaign_order():
    cells = run_campaign_module.parse_campaign(_REAL_CAMPAIGN)
    assert run_campaign_module.shard_cells(cells, 0, 1) == cells


def test_main_shard_flag_partitions_and_skip_complete_still_applies(tmp_path, capsys):
    campaign, groups = _campaign_with_bonus(tmp_path)
    runs_dir = tmp_path / "runs"
    # Complete the pilot cell so whichever shard holds it reports a skip.
    for seed in range(2):
        _completed_run(runs_dir, f"done-{seed}", order=32, index=18, width=64, seed=seed)
    seen: list[str] = []
    for i in range(2):
        code = run_campaign_module.main(
            [
                "--campaign",
                str(campaign),
                "--group-properties",
                str(groups),
                "--runs-dir",
                str(runs_dir),
                "--include-bonus",
                "--shard",
                f"{i}/2",
                "--dry-run",
            ]
        )
        assert code == 0
        seen.append(capsys.readouterr().out)
    combined = "".join(seen)
    # The two shards together cover all four cells exactly once, and the
    # completed pilot cell is a SKIP in whichever shard it landed.
    for marker in (
        "SmallGroup(32,18)",
        "SmallGroup(27,3)",
        "SmallGroup(64,92)",
        "SmallGroup(128,197)",
    ):
        assert combined.count(f"{marker} ") == 1
    assert "SKIP (already complete)" in combined


def test_main_rejects_a_malformed_shard(tmp_path, capsys):
    campaign, groups = _campaign_with_bonus(tmp_path)
    code = run_campaign_module.main(
        [
            "--campaign",
            str(campaign),
            "--group-properties",
            str(groups),
            "--runs-dir",
            str(tmp_path / "runs"),
            "--shard",
            "8/8",
            "--dry-run",
        ]
    )
    assert code == 1
    assert "--shard error" in capsys.readouterr().err


def test_main_empty_shard_exits_zero(tmp_path, capsys):
    # 2 non-bonus cells over 5 workers: at least one shard is empty.
    campaign, groups = _campaign_with_bonus(tmp_path)
    code = run_campaign_module.main(
        [
            "--campaign",
            str(campaign),
            "--group-properties",
            str(groups),
            "--runs-dir",
            str(tmp_path / "runs"),
            "--shard",
            "4/5",
            "--dry-run",
        ]
    )
    assert code == 0
    assert "this shard is empty" in capsys.readouterr().out


# --- the checked-in campaign file ---------------------------------------------------


def test_checked_in_core_campaign_parses_with_the_pre_registered_shape():
    """Bonus phases must never change the pre-registered shape
    (docs/core-study.md's group table and run plan, as revised by the
    2026-07-20 restart in internal/docs/handoff.md)."""
    cells = run_campaign_module.parse_campaign(_REAL_CAMPAIGN)
    registered = [cell for cell in cells if cell.phase not in run_campaign_module.BONUS_PHASES]
    pilot = [cell for cell in cells if cell.phase == "pilot"]
    core = [cell for cell in cells if cell.phase == "core"]
    extra = [cell for cell in cells if cell.phase == "extra"]

    assert len(registered) == 35
    assert len(pilot) == 1 and len(core) == 31 and len(extra) == 3
    # the pre-registered phases come first, pilot then core then extra, contiguous
    assert [cell.phase for cell in registered] == ["pilot"] * 1 + ["core"] * 31 + ["extra"] * 3
    # pilot: one D32 smoke cell, width 128, 2 seeds (no fresh phase-0 re-run)
    assert pilot[0].group_key == (32, 18)
    assert pilot[0].width == 128
    assert pilot[0].seed_range == range(0, 2)
    # core: 26 distinct groups -- 5 of them (the D32/QD32/Q32 case study and
    # the GL(2,3)/(48,28) cocycle pair) run at both width 128 and 256, so
    # (order, index) is not unique within core, only group_key membership is
    assert len({cell.group_key for cell in core}) == 26
    assert all(cell.seed_range == range(0, 50) for cell in core)
    # extra: the D5 additions, width 128, 50 seeds
    assert {cell.group_key for cell in extra} == {(81, 7), (192, 10), (192, 24)}
    assert all(cell.width == 128 and cell.seed_range == range(0, 50) for cell in extra)
    # 29 distinct pre-registered groups; 1,702 pre-registered models
    assert len({cell.group_key for cell in registered}) == 29
    assert sum(len(cell.seed_range) for cell in registered) == 1702
    # cheapest-first within each phase (ascending group order)
    for block in (pilot, core, extra):
        orders = [cell.order for cell in block]
        assert orders == sorted(orders)
    # every cell runs the pre-registered experiment preset
    assert all(cell.experiment == "core" for cell in cells)


def test_checked_in_bonus_phases_have_the_commissioned_shape():
    """Bonus cells (E6 + E2) are appended after every pre-registered cell and
    never count toward the pre-registered totals."""
    cells = run_campaign_module.parse_campaign(_REAL_CAMPAIGN)
    e6 = [cell for cell in cells if cell.phase == "bonus-e6"]
    e2 = [cell for cell in cells if cell.phase == "bonus-e2"]
    extra = [cell for cell in cells if cell.phase == "extra"]

    assert len(e6) == 6 and len(e2) == 20
    assert len(cells) == 61  # 35 pre-registered (pilot+core+extra) + 26 bonus
    # bonus comes last: e6 then e2, contiguous
    assert [cell.phase for cell in cells[35:]] == ["bonus-e6"] * 6 + ["bonus-e2"] * 20
    assert {cell.group_key for cell in e6} == {
        (64, 92),
        (64, 93),
        (64, 99),
        (64, 100),
        (192, 397),
        (192, 584),
    }
    assert {cell.group_key for cell in e2} == {
        (128, index)
        for index in (
            197,
            198,
            280,
            281,
            288,
            290,
            372,
            378,
            451,
            452,
            602,
            634,
            635,
            641,
            642,
            651,
            652,
            675,
            676,
            687,
        )
    }
    assert all(cell.width == 128 and cell.seed_range == range(0, 50) for cell in e6 + e2)
    # bonus groups never overlap the pre-registered 29
    registered_groups = {
        cell.group_key for cell in cells if cell.phase not in run_campaign_module.BONUS_PHASES
    }
    assert registered_groups.isdisjoint({cell.group_key for cell in e6 + e2})
    # the extra phase's D5 additions, width 128, 50 seeds, never overlapping
    # pilot/core/bonus groups
    assert {cell.group_key for cell in extra} == {(81, 7), (192, 10), (192, 24)}
    assert all(cell.width == 128 and cell.seed_range == range(0, 50) for cell in extra)
    non_extra_groups = {cell.group_key for cell in cells if cell.phase != "extra"}
    assert {(81, 7), (192, 10), (192, 24)}.isdisjoint(non_extra_groups)
    # whole-file totals with bonus included
    assert len({cell.group_key for cell in cells}) == 55
    assert sum(len(cell.seed_range) for cell in cells) == 3002


@pytest.mark.skipif(
    not _REAL_GROUP_PROPERTIES.is_file(),
    reason="data/group_properties_full.jsonl is gitignored and absent here",
)
def test_checked_in_core_campaign_groups_all_exist_in_the_real_catalogue():
    cells = run_campaign_module.parse_campaign(_REAL_CAMPAIGN)
    known = run_campaign_module.load_known_groups(_REAL_GROUP_PROPERTIES)
    assert run_campaign_module.validate_cells(cells, known) == []
