"""Dip-aware checkpoint selection (instruments/checkpoints.py).

The rule under test, from the study's dip-handling decision: runs whose
resolved config carries ``snapshot.final_window_epochs`` select the last
final-window epoch whose ``run.log`` row clears ``val/accuracy >= 0.99``
(falling back to the nearest stable trajectory snapshot when the whole window
is dipped); older runs gate ``final.pt`` on its own final ``run.log`` row and
substitute the nearest stable trajectory snapshot when dipped. Every
substitution is recorded as data.

The run fixtures are real short training runs (they produce genuine snapshot
and final-window files through the production code path); ``run.log`` is then
rewritten with controlled accuracy patterns, because the selection rule reads
stability from ``run.log`` alone and a 12-epoch model never reaches 0.99 on
its own. A smoke test against a real salvaged campaign ``run.log`` covers the
parsing path on production data (skipped when ``results-archive/`` is not
present, e.g. in CI).
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from group_algorithm_interp.config import ExperimentConfig, LoggingConfig, ProjectConfig
from group_algorithm_interp.experiment import GroupGeneralizationExperiment
from group_algorithm_interp.instruments.checkpoints import (
    parse_run_log,
    select_checkpoint,
)

_ARCHIVE_RUN = (
    Path(__file__).resolve().parent.parent
    / "results-archive"
    / "2026-07-15_125014_165786_core_7d2298"
)

EPOCHS = 12
WINDOW = 3  # final-window epochs 9, 10, 11
STEP_EPOCHS = (0, 1, 2, 4, 8)  # dense powers of two to 4, then interval 4


def _run(runs_root: Path, *, window: int, name: str) -> Path:
    config = ProjectConfig(
        device="cpu",
        seed=0,
        data={"group": "C4", "train_frac": 0.5, "split_seed": 0},
        model={"d_model": 16, "d_mlp": 32, "n_heads": 1},
        optim={"epochs": EPOCHS, "log_every": 1, "print_every": 1},
        snapshot={
            "enabled": True,
            "interval": 4,
            "log_dense_until": 4,
            "event_based": False,
            "final_window_epochs": window,
        },
        logging=LoggingConfig(mode="disabled"),
        experiment=ExperimentConfig(name=name),
    )
    experiment = GroupGeneralizationExperiment(config, runs_root=runs_root)
    experiment.execute()
    return experiment.run_dir


@pytest.fixture(scope="module")
def window_run(tmp_path_factory) -> Path:
    """A real run with final-window snapshots (the restarted-campaign layout)."""
    return _run(tmp_path_factory.mktemp("window-run"), window=WINDOW, name="ckpt-window")


@pytest.fixture(scope="module")
def old_style_run(tmp_path_factory) -> Path:
    """A real run mimicking the terminated campaign: no final-window files and
    no ``final_window_epochs`` key in its resolved config (the field postdates
    those runs, so it is stripped from the YAML, not just zeroed)."""
    run_dir = _run(tmp_path_factory.mktemp("old-run"), window=0, name="ckpt-old")
    resolved = run_dir / "resolved_config.yaml"
    lines = [
        line for line in resolved.read_text().splitlines() if "final_window_epochs" not in line
    ]
    resolved.write_text("\n".join(lines) + "\n")
    return run_dir


def _write_log(run_dir: Path, accuracies: dict[int, float | str]) -> None:
    """Rewrite run.log with controlled val/accuracy values in the ensemble
    path's bare dialect (the parser must accept both dialects; the fixtures'
    original logs are the single-path dialect). A string value is written
    verbatim (used to plant a bare ``nan``)."""
    lines = []
    for epoch, accuracy in sorted(accuracies.items()):
        lines.append(
            f"epoch {epoch} | {{'train/loss': 0.001, 'val/accuracy': {accuracy}, "
            f"'val/unleaked_accuracy': {accuracy}, 'val/weight_norm': 25.0}}"
        )
    (run_dir / "run.log").write_text("\n".join(lines) + "\n")


def _full_log(**overrides: float | str) -> dict[int, float | str]:
    """All EPOCHS epochs stable at 0.995, with per-epoch overrides."""
    log: dict[int, float | str] = {epoch: 0.995 for epoch in range(EPOCHS)}
    for key, value in overrides.items():
        log[int(key.lstrip("e"))] = value
    return log


# ---------------------------------------------------------------------------
# run.log parsing
# ---------------------------------------------------------------------------


def test_parse_run_log_single_path_dialect(window_run):
    """The production single-seed run.log (timestamp/level-prefixed lines)
    parses into one row per evaluated epoch with the real metric keys."""
    rows = parse_run_log(window_run / "run.log")
    assert [row.epoch for row in rows] == list(range(EPOCHS))
    assert set(rows[0].metrics) >= {"val/accuracy", "val/unleaked_accuracy", "train/loss"}
    assert 0.0 <= rows[0].metrics["val/accuracy"] <= 1.0


def test_parse_run_log_ensemble_dialect_and_nan(tmp_path):
    """Bare `epoch N | {...}` lines parse too, and a bare `nan` (the degenerate
    unleaked-subset case) becomes float NaN rather than a crash."""
    run_dir = tmp_path
    _write_log(run_dir, {0: 0.5, 3: "nan", 7: 1.0})
    rows = parse_run_log(run_dir / "run.log")
    assert [row.epoch for row in rows] == [0, 3, 7]
    assert rows[0].metrics["val/accuracy"] == 0.5
    assert math.isnan(rows[1].metrics["val/accuracy"])
    assert rows[2].metrics["val/accuracy"] == 1.0


# ---------------------------------------------------------------------------
# Final-window rule (runs whose resolved config carries the field)
# ---------------------------------------------------------------------------


def test_window_rule_selects_last_stable_window_epoch(window_run):
    _write_log(window_run, _full_log())
    selection = select_checkpoint(window_run)
    assert selection.rule == "final_window"
    assert selection.path is not None and selection.path.name == "final_epoch_11.pt"
    assert selection.epoch == 11
    assert selection.metric_value == 0.995
    assert selection.substitution is None
    assert selection.rejected == []


def test_window_rule_records_dipped_final_epoch_and_selects_earlier(window_run):
    _write_log(window_run, _full_log(e11=0.859))
    selection = select_checkpoint(window_run)
    assert selection.path is not None and selection.path.name == "final_epoch_10.pt"
    assert selection.epoch == 10
    assert selection.substitution == "window"
    assert selection.rejected == [{"checkpoint": "final_epoch_11.pt", "epoch": 11, "value": 0.859}]


def test_window_rule_falls_back_to_nearest_stable_trajectory_snapshot(window_run):
    """All three window epochs dipped -> the latest stable step snapshot
    substitutes (epoch 8 here), and every rejected candidate is recorded."""
    _write_log(window_run, _full_log(e9=0.9, e10=0.9, e11=0.9))
    selection = select_checkpoint(window_run)
    assert selection.path is not None and selection.path.name == "step_8.pt"
    assert selection.epoch == 8
    assert selection.substitution == "trajectory"
    assert [r["epoch"] for r in selection.rejected] == [11, 10, 9]


def test_window_epoch_without_log_row_is_unassessable(window_run):
    """A snapshot with no run.log row at its own epoch cannot be verified
    stable; it is passed over and recorded with value None."""
    log = _full_log()
    del log[11]
    _write_log(window_run, log)
    selection = select_checkpoint(window_run)
    assert selection.path is not None and selection.path.name == "final_epoch_10.pt"
    assert selection.substitution == "window"
    assert selection.rejected == [{"checkpoint": "final_epoch_11.pt", "epoch": 11, "value": None}]


def test_nan_metric_counts_as_dipped(window_run):
    _write_log(window_run, _full_log(e11="nan"))
    selection = select_checkpoint(window_run)
    assert selection.path is not None and selection.path.name == "final_epoch_10.pt"
    rejected_value = selection.rejected[0]["value"]
    assert rejected_value is not None and math.isnan(rejected_value)


# ---------------------------------------------------------------------------
# Final-gate rule (older runs, field absent from the resolved config)
# ---------------------------------------------------------------------------


def test_old_run_final_gate_accepts_stable_final(old_style_run):
    _write_log(old_style_run, _full_log())
    selection = select_checkpoint(old_style_run)
    assert selection.rule == "final_gate"
    assert selection.path is not None and selection.path.name == "final.pt"
    assert selection.epoch == EPOCHS - 1
    assert selection.substitution is None


def test_old_run_dipped_final_substitutes_nearest_stable_snapshot(old_style_run):
    _write_log(old_style_run, _full_log(e11=0.85))
    selection = select_checkpoint(old_style_run)
    assert selection.path is not None and selection.path.name == "step_8.pt"
    assert selection.epoch == 8
    assert selection.substitution == "trajectory"
    assert selection.rejected[0] == {"checkpoint": "final.pt", "epoch": 11, "value": 0.85}


def test_no_stable_checkpoint_selects_nothing(old_style_run):
    """A run that never clears the bar anywhere (a censored seed) selects no
    checkpoint: path None with the reason recorded -- reported as data, never
    silently analysed."""
    _write_log(old_style_run, {epoch: 0.5 for epoch in range(EPOCHS)})
    selection = select_checkpoint(old_style_run)
    assert selection.path is None
    assert selection.epoch is None
    assert selection.reason is not None and "no checkpoint" in selection.reason
    # final.pt plus every trajectory snapshot was examined and rejected.
    assert {r["checkpoint"] for r in selection.rejected} == {
        "final.pt",
        *(f"step_{epoch}.pt" for epoch in STEP_EPOCHS),
    }


def test_threshold_and_metric_are_parameters(old_style_run):
    """The bar and the metric are explicit parameters recorded in the
    selection, not constants baked into the rule."""
    _write_log(old_style_run, {epoch: 0.5 for epoch in range(EPOCHS)})
    selection = select_checkpoint(old_style_run, metric="val/unleaked_accuracy", threshold=0.4)
    assert selection.path is not None and selection.path.name == "final.pt"
    assert selection.metric == "val/unleaked_accuracy"
    assert selection.threshold == 0.4
    record = selection.to_record()
    assert record["metric"] == "val/unleaked_accuracy"
    assert record["threshold"] == 0.4
    assert record["checkpoint"] == "final.pt"


def test_missing_run_log_selects_nothing(tmp_path, old_style_run):
    import shutil

    clone = tmp_path / "clone"
    shutil.copytree(old_style_run, clone)
    (clone / "run.log").unlink()
    selection = select_checkpoint(clone)
    assert selection.path is None
    assert selection.reason is not None and "run.log" in selection.reason


# ---------------------------------------------------------------------------
# Real salvaged campaign run (parsing-path smoke on production data)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _ARCHIVE_RUN.is_dir(), reason="results-archive/ not present")
def test_real_salvaged_run_log_parses_and_gates_final():
    """One salvaged run from the terminated campaign: 30k per-epoch rows in the
    ensemble dialect, a final row below the bar (val/accuracy ~0.93), no
    surviving trajectory snapshots -- so the final-gate rule must reject
    final.pt and select nothing, recording why."""
    rows = parse_run_log(_ARCHIVE_RUN / "run.log")
    assert len(rows) == 30_000
    assert rows[-1].epoch == 29_999
    assert rows[-1].metrics["val/accuracy"] == pytest.approx(0.9317073, abs=1e-6)

    selection = select_checkpoint(_ARCHIVE_RUN)
    assert selection.rule == "final_gate"  # its resolved config predates the window field
    assert selection.path is None
    assert selection.rejected[0]["checkpoint"] == "final.pt"
    assert selection.rejected[0]["value"] == pytest.approx(0.9317073, abs=1e-6)
