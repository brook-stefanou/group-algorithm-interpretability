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

import gzip
import json
import math
from pathlib import Path

import pytest

from group_algorithm_interp.config import ExperimentConfig, LoggingConfig, ProjectConfig
from group_algorithm_interp.experiment import GroupGeneralizationExperiment
from group_algorithm_interp.instruments.checkpoints import (
    parse_run_log,
    resolve_checkpoint,
    resolve_run_log,
    run_log_rows,
    select_checkpoint,
    selection_from_json,
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


# ---------------------------------------------------------------------------
# Curated shipped layout (flat checkpoints + selection.json; log gzipped/absent)
#
# The ship hook prunes each run into a curated layout the instruments must read
# directly: checkpoints FLAT at the run-dir root (no ``checkpoints/``), the log
# gzipped as ``run.log.gz`` or gone entirely, and the dip-aware pick recorded in
# ``selection.json`` under one of two shapes. select_checkpoint takes the pick
# straight from selection.json rather than recomputing over files that no longer
# exist in that layout.
# ---------------------------------------------------------------------------


def _curated_run(run_dir: Path, *, selection: dict, window: int = 5) -> Path:
    """A minimal curated run dir: a resolved config (so the rule can be named),
    a ``selection.json``, and flat stub checkpoints for every filename the
    selection references. No ``checkpoints/`` dir and no ``run.log``."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.yaml").write_text(f"snapshot:\n  final_window_epochs: {window}\n")
    (run_dir / "selection.json").write_text(json.dumps(selection))
    names: set[str] = set()

    def _collect(node: object) -> None:
        if isinstance(node, dict):
            filename = node.get("checkpoint") or node.get("filename")
            if isinstance(filename, str):
                names.add(filename)
            for value in node.values():
                _collect(value)

    _collect(selection)
    for name in names:
        (run_dir / name).write_bytes(b"stub")
    return run_dir


def _selections_stable_end(checkpoint: str | None, *, reason: str | None = None) -> dict:
    """Flat-shape selection.json: ``selections.stable_end`` is a full
    CheckpointSelection.to_record()."""
    return {
        "selections": {
            "stable_end": {
                "rule": "final_window",
                "metric": "val/accuracy",
                "threshold": 0.99,
                "checkpoint": checkpoint,
                "epoch": 11 if checkpoint else None,
                "metric_value": 1.0 if checkpoint else None,
                "substitution": None,
                "rejected": [],
                "reason": reason,
            }
        }
    }


def _categories_stable_end(entry: dict | None, *, anomalies: list[str] | None = None) -> dict:
    """ship_runs.py-shape selection.json: ``categories.stable_end`` is a lighter
    curated entry (or None)."""
    return {
        "threshold": 0.99,
        "leaked_metric_key": "val/accuracy",
        "categories": {"stable_end": entry},
        "anomalies": anomalies or [],
    }


def test_curated_selections_schema_grokked_takes_recorded_pick(tmp_path):
    run = _curated_run(tmp_path / "sel-grok", selection=_selections_stable_end("final_epoch_11.pt"))
    selection = select_checkpoint(run)
    assert selection.path is not None and selection.path.name == "final_epoch_11.pt"
    assert selection.path.parent == run  # resolved FLAT, not under checkpoints/
    assert selection.epoch == 11
    assert selection.metric_value == 1.0
    assert selection.rule == "final_window"
    assert selection.substitution is None
    assert selection.reason is None


def test_curated_selections_schema_censored_selects_nothing(tmp_path):
    run = _curated_run(
        tmp_path / "sel-cens",
        selection=_selections_stable_end(None, reason="a censored or never-stable run"),
    )
    selection = select_checkpoint(run)
    assert selection.path is None
    assert selection.epoch is None
    assert selection.reason == "a censored or never-stable run"


def test_curated_categories_schema_grokked_fills_missing_fields(tmp_path):
    entry = {
        "epoch": 11,
        "metric_key": "val/accuracy",
        "metric_value": 0.995,
        "filename": "step_11.pt",
    }
    run = _curated_run(tmp_path / "cat-grok", selection=_categories_stable_end(entry))
    selection = select_checkpoint(run)
    assert selection.path is not None and selection.path.name == "step_11.pt"
    assert selection.epoch == 11
    assert selection.metric_value == 0.995
    assert selection.metric == "val/accuracy"
    assert selection.threshold == 0.99
    assert selection.rule == "final_window"  # inferred from the resolved config


def test_curated_categories_schema_censored_uses_anomaly_reason(tmp_path):
    run = _curated_run(
        tmp_path / "cat-cens",
        selection=_categories_stable_end(
            None, anomalies=["stable_end: no checkpoint clears the bar"]
        ),
    )
    selection = select_checkpoint(run)
    assert selection.path is None
    assert selection.reason == "stable_end: no checkpoint clears the bar"


def test_curated_run_needs_no_run_log(tmp_path):
    """The 288 earliest shipped runs kept no log at all; the pick still resolves
    from selection.json, and the log resolvers degrade cleanly."""
    run = _curated_run(tmp_path / "nolog", selection=_selections_stable_end("final_epoch_11.pt"))
    assert not (run / "run.log").exists() and not (run / "run.log.gz").exists()
    assert resolve_run_log(run) is None
    assert run_log_rows(run) == []
    selection = select_checkpoint(run)
    assert selection.path is not None and selection.path.name == "final_epoch_11.pt"


def test_curated_checkpoint_named_but_missing_on_disk(tmp_path):
    run = tmp_path / "missing"
    run.mkdir()
    (run / "resolved_config.yaml").write_text("snapshot:\n  final_window_epochs: 5\n")
    (run / "selection.json").write_text(json.dumps(_selections_stable_end("gone.pt")))
    # Deliberately do not write gone.pt.
    selection = select_checkpoint(run)
    assert selection.path is None
    assert selection.reason is not None and "not on disk" in selection.reason


def test_selection_from_json_absent_returns_none(tmp_path):
    run = tmp_path / "plain"
    run.mkdir()
    (run / "resolved_config.yaml").write_text("snapshot:\n  final_window_epochs: 5\n")
    assert selection_from_json(run) is None


# ---------------------------------------------------------------------------
# Gzipped log reading + flat/nested checkpoint resolution
# ---------------------------------------------------------------------------


def test_parse_run_log_reads_gzip(tmp_path):
    """The ship hook gzips the log; parse_run_log reads run.log.gz transparently."""
    text = "epoch 0 | {'val/accuracy': 0.5}\nepoch 1 | {'val/accuracy': 0.9}\n"
    gz_path = tmp_path / "run.log.gz"
    with gzip.open(gz_path, "wt") as handle:
        handle.write(text)
    rows = parse_run_log(gz_path)
    assert [row.epoch for row in rows] == [0, 1]
    assert rows[1].metrics["val/accuracy"] == 0.9


def test_resolve_run_log_prefers_plain_then_gz(tmp_path):
    run = tmp_path / "logs"
    run.mkdir()
    assert resolve_run_log(run) is None
    gz = run / "run.log.gz"
    with gzip.open(gz, "wt") as handle:
        handle.write("epoch 0 | {'val/accuracy': 1.0}\n")
    assert resolve_run_log(run) == gz
    plain = run / "run.log"
    plain.write_text("epoch 0 | {'val/accuracy': 1.0}\n")
    assert resolve_run_log(run) == plain  # plain wins when both are present
    assert len(run_log_rows(run)) == 1


def test_resolve_checkpoint_flat_first_then_nested(tmp_path):
    run = tmp_path / "ckpts"
    run.mkdir()
    assert resolve_checkpoint(run, "final.pt") is None
    assert resolve_checkpoint(run, None) is None
    nested = run / "checkpoints"
    nested.mkdir()
    (nested / "final.pt").write_bytes(b"old")
    assert resolve_checkpoint(run, "final.pt") == nested / "final.pt"  # old training layout
    flat = run / "final.pt"
    flat.write_bytes(b"new")
    assert resolve_checkpoint(run, "final.pt") == flat  # curated layout wins when both exist


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
