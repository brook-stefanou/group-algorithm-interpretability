"""Focused tests for `scripts/stream_runs.py` (the live-observability sidecar).

Offline and network-free: the W&B boundary is the sidecar's own injectable
``WandbSink`` seam, replaced here by a recording fake; run directories are
synthesised on disk (plus one real short single-seed training run for the
live-occupancy path). The script is an executable entry point, not part of the
installed package, so it is loaded by file path (mirrors
``tests/test_sync_runs.py``).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

from group_algorithm_interp.config import ExperimentConfig, LoggingConfig, ProjectConfig
from group_algorithm_interp.experiment import GroupGeneralizationExperiment

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "stream_runs.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("stream_runs", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["stream_runs"] = module
    spec.loader.exec_module(module)
    return module


stream_runs = _load_script()


class FakeSink:
    """Records every sink call; failures and resume state are injectable."""

    def __init__(self) -> None:
        self.init_specs: list[Any] = []
        self.logs: dict[str, list[dict[str, float]]] = {}
        self.summaries: dict[str, dict[str, Any]] = {}
        self.finished: list[str] = []
        self.resume_epochs: dict[str, int] = {}
        self.complete_ids: set[str] = set()
        self.fail_next_logs = 0

    def init_run(self, spec: Any) -> str:
        self.init_specs.append(spec)
        return str(spec.run_id)

    def last_epoch(self, handle: str) -> int:
        return self.resume_epochs.get(handle, -1)

    def stream_complete(self, handle: str) -> bool:
        return handle in self.complete_ids

    def log(self, handle: str, payload: dict[str, float]) -> None:
        if self.fail_next_logs > 0:
            self.fail_next_logs -= 1
            raise RuntimeError("simulated W&B outage")
        self.logs.setdefault(handle, []).append(payload)

    def set_summary(self, handle: str, mapping: dict[str, Any]) -> None:
        self.summaries.setdefault(handle, {}).update(mapping)

    def finish(self, handle: str) -> None:
        self.finished.append(handle)


def _make_run_dir(
    root: Path,
    run_id: str,
    *,
    status: str = "running",
    order: int = 8,
    index: int = 3,
    width: int = 32,
    seed: int = 3,
) -> Path:
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "run_id": run_id,
                "status": status,
                "dataset": {
                    "name": f"SmallGroup({order},{index})",
                    "leakage": {"generalize_metric": "unleaked_accuracy"},
                    "split_seed": 0,
                    "spec_hash": "dataset-hash-abc",
                },
                "provenance": {
                    "config_hash": "config-hash-abc",
                    "config_group_hash": "group-hash-abc",
                    "campaign_id": "campaign-2026-07-20",
                },
            }
        )
    )
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(
            {
                "seed": seed,
                "data": {"group": {"order": order, "index": index}},
                "model": {"d_model": width},
                "optim": {"generalize_test_acc": 0.99, "generalize_patience": 3},
                "experiment": {"name": "core"},
            }
        )
    )
    return run_dir


def _set_status(run_dir: Path, status: str) -> None:
    manifest = yaml.safe_load((run_dir / "manifest.yaml").read_text())
    manifest["status"] = status
    (run_dir / "manifest.yaml").write_text(yaml.safe_dump(manifest))


def _epoch_line(epoch: int, unleaked: float = 0.5) -> str:
    metrics = {
        "train/loss": 1.0,
        "train/accuracy": 0.9,
        "val/loss": 1.2,
        "val/accuracy": 0.8,
        "val/unleaked_accuracy": unleaked,
        "val/weight_norm": 10.0,
    }
    return f"epoch {epoch} | {metrics}"


def _sidecar(root: Path, sink: FakeSink, **kwargs: Any) -> Any:
    return stream_runs.Sidecar(root, sink, **kwargs)


# ---------------------------------------------------------------------------
# parsing and naming
# ---------------------------------------------------------------------------


def test_parse_epoch_line_accepts_both_dialects():
    bare = stream_runs.parse_epoch_line("epoch 7 | {'val/accuracy': 0.5}")
    prefixed = stream_runs.parse_epoch_line(
        "12:00:00 | INFO | group_algorithm_interp | epoch 7 | {'val/accuracy': 0.5}"
    )
    assert bare == (7, {"val/accuracy": 0.5})
    assert prefixed == (7, {"val/accuracy": 0.5})


def test_parse_metrics_handles_nonfinite_tokens():
    parsed = stream_runs.parse_metrics("{'val/unleaked_accuracy': nan, 'val/loss': inf}")
    assert parsed["val/loss"] == float("inf")
    assert parsed["val/unleaked_accuracy"] != parsed["val/unleaked_accuracy"]  # NaN


def test_parse_completed_line_accepts_both_shapes():
    flat = stream_runs.parse_completed_line("completed run-abc | {'val/accuracy': 1.0}")
    assert flat == {"val/accuracy": 1.0}
    nested = stream_runs.parse_completed_line(
        "completed run-abc | {'completed_steps': 90, 'metrics': {'val/accuracy': 1.0}}"
    )
    assert nested == {"completed_steps": 90.0, "val/accuracy": 1.0}
    assert stream_runs.parse_completed_line("something else") is None


def test_build_run_spec_uses_campaign_cell_naming(tmp_path):
    run_dir = _make_run_dir(tmp_path, "run-a")
    lookup = {
        (8, 3, 32): {
            "order": 8,
            "index": 3,
            "width": 32,
            "name": "D4",
            "phase": "core",
            "note": "C1 tier-1 pair with (8,4); the dihedral/quaternion contrast",
        }
    }
    spec = stream_runs.build_run_spec(run_dir, lookup)
    assert spec.run_id == "run-a"
    assert spec.name == "D4 (8,3) w32 s03"  # zero-padded seed
    assert spec.tags == ["core", "D4", "campaign-v2"]
    assert spec.wandb_group == "group-hash-abc"
    assert spec.config["raw_run_id"] == "run-a"
    assert spec.config["cell_name"] == "D4"
    assert spec.config["group_canonical_name"] == "SmallGroup(8,3)"
    assert spec.config["config_hash"] == "config-hash-abc"
    assert spec.config["config_group_hash"] == "group-hash-abc"
    assert spec.config["campaign_id"] == "campaign-2026-07-20"
    assert spec.config["split_seed"] == 0
    assert spec.config["dataset_spec_hash"] == "dataset-hash-abc"
    assert spec.config["pair_partner_order"] == 8
    assert spec.config["pair_partner_index"] == 4
    assert spec.generalize_metric_key == "val/unleaked_accuracy"
    assert spec.generalize_test_acc == 0.99
    assert spec.generalize_patience == 3


def test_build_run_spec_without_a_cell_match(tmp_path):
    run_dir = _make_run_dir(tmp_path, "run-b")
    spec = stream_runs.build_run_spec(run_dir, {})
    assert spec.name == "(8,3) (8,3) w32 s03"
    assert spec.config["phase"] == "unknown"
    assert spec.config["pair_partner_order"] is None
    assert spec.config["pair_partner_index"] is None
    assert spec.tags == ["unknown", "SmallGroup(8,3)", "campaign-v2"]


def test_load_campaign_lookup_missing_file(tmp_path):
    assert stream_runs.load_campaign_lookup(tmp_path / "nope.yaml") == {}


def test_load_campaign_lookup_malformed_yaml_degrades(tmp_path):
    path = tmp_path / "core.yaml"
    path.write_text("cells: [\n  - order: 4\n")  # unbalanced flow sequence
    assert stream_runs.load_campaign_lookup(path) == {}


def test_load_campaign_lookup_non_mapping_top_level_degrades(tmp_path, capsys):
    path = tmp_path / "core.yaml"
    path.write_text("- order: 4\n  index: 1\n  width: 16\n")
    assert stream_runs.load_campaign_lookup(path) == {}
    assert "not a mapping" in capsys.readouterr().err


def test_load_campaign_lookup_skips_malformed_cell_entry(tmp_path, capsys):
    path = tmp_path / "core.yaml"
    path.write_text(
        """
cells:
  - order: 4
    index: 1
    name: "C4-bad-no-width"
  - order: 8
    index: 3
    width: 32
    name: "D4-good"
"""
    )
    lookup = stream_runs.load_campaign_lookup(path)
    assert (8, 3, 32) in lookup
    assert (4, 1, 16) not in lookup
    assert "malformed campaign cell entry" in capsys.readouterr().err


def test_sanitise_run_id():
    assert stream_runs.sanitise_run_id("run/with:odd chars") == "run-with-odd-chars"


# ---------------------------------------------------------------------------
# pair-partner parsing and seed formatting
# ---------------------------------------------------------------------------


def test_parse_pair_partner_finds_the_declared_partner():
    assert stream_runs.parse_pair_partner(
        "w256 probe -- 3^{1+2}+, C1 tier-1 pair with (27,4); grokked 0/52 at 30k/w128"
    ) == (27, 4)
    assert stream_runs.parse_pair_partner(
        "E2 order-128 clean pair with (128,198) (extension supply); runs only with --include-bonus"
    ) == (128, 198)


def test_parse_pair_partner_takes_the_first_mentioned_partner_in_a_multiway_note():
    # The D32/QD32/Q32 case study: not individually parenthesised.
    assert stream_runs.parse_pair_partner(
        "C2 case study (with 32,19 and 32,20) -- coset route available"
    ) == (32, 19)


def test_parse_pair_partner_returns_none_when_no_partner_is_declared():
    assert stream_runs.parse_pair_partner(None) is None
    assert stream_runs.parse_pair_partner("smoke cell -- must complete before core starts") is None
    assert (
        stream_runs.parse_pair_partner("C3 anchor -- calibrates every reading, pooled with nothing")
        is None
    )
    assert (
        stream_runs.parse_pair_partner(
            "E6 bundle (extension supply); runs only with --include-bonus"
        )
        is None
    )


def test_format_seed_zero_pads_so_names_sort_correctly():
    assert stream_runs.format_seed(2) == "02"
    assert stream_runs.format_seed(10) == "10"
    assert stream_runs.format_seed(49) == "49"
    assert "02" < "10"  # the point of the padding: lexicographic order matches numeric order


# ---------------------------------------------------------------------------
# the WandbSink boundary (the one place that touches the real SDK)
# ---------------------------------------------------------------------------


def test_wandb_sink_init_run_passes_the_pooling_group():
    """``group`` must reach ``wandb.init`` so every seed of one cell/width
    pools together in the UI -- the field the live sidecar previously left
    unset (unlike the occupancy publisher)."""

    class _FakeRun:
        def define_metric(self, *_args: Any, **_kwargs: Any) -> None:
            pass

    class _FakeSettings:
        def __init__(self, **_kwargs: Any) -> None:
            pass

    class _FakeWandbModule:
        def __init__(self) -> None:
            self.init_kwargs: dict[str, Any] = {}
            self.Settings = _FakeSettings

        def init(self, **kwargs: Any) -> _FakeRun:
            self.init_kwargs = kwargs
            return _FakeRun()

    sink = stream_runs.WandbSink.__new__(stream_runs.WandbSink)
    fake_module = _FakeWandbModule()
    sink._wandb = fake_module  # type: ignore[attr-defined]
    sink.project = "test-project"
    sink.entity = None
    spec = stream_runs.RunSpec(
        run_id="run-a",
        name="D4 (8,3) w32 s03",
        config={},
        tags=["core"],
        wandb_group="group-hash-abc",
        generalize_metric_key="val/unleaked_accuracy",
        generalize_test_acc=0.99,
        generalize_patience=3,
    )
    sink.init_run(spec)
    assert fake_module.init_kwargs["group"] == "group-hash-abc"
    assert fake_module.init_kwargs["id"] == "run-a"
    assert fake_module.init_kwargs["name"] == "D4 (8,3) w32 s03"


# ---------------------------------------------------------------------------
# tailing
# ---------------------------------------------------------------------------


def test_log_tail_is_incremental_and_holds_partial_lines(tmp_path):
    path = tmp_path / "run.log"
    tail = stream_runs.LogTail(path)
    assert tail.read_new_lines() == []  # file absent

    path.write_text(_epoch_line(0) + "\n" + "epoch 1 | {'val/a")
    lines = tail.read_new_lines()
    assert len(lines) == 1 and lines[0].startswith("epoch 0")

    with path.open("a") as handle:
        handle.write("ccuracy': 0.5}\n")
    lines = tail.read_new_lines()
    assert lines == ["epoch 1 | {'val/accuracy': 0.5}"]
    assert tail.read_new_lines() == []


def test_log_tail_restarts_after_truncation(tmp_path):
    path = tmp_path / "run.log"
    tail = stream_runs.LogTail(path)
    path.write_text(_epoch_line(0) + "\n" + _epoch_line(1) + "\n")
    assert len(tail.read_new_lines()) == 2
    path.write_text(_epoch_line(2) + "\n")  # rewritten shorter
    lines = tail.read_new_lines()
    assert len(lines) == 1 and lines[0].startswith("epoch 2")


# ---------------------------------------------------------------------------
# grok tracking
# ---------------------------------------------------------------------------


def test_grok_tracker_finds_streak_start():
    tracker = stream_runs.GrokTracker(
        metric_key="val/unleaked_accuracy", threshold=0.99, patience=3
    )
    for epoch in range(5):
        tracker.observe(epoch, {"val/unleaked_accuracy": 0.5})
    tracker.observe(5, {"val/unleaked_accuracy": 1.0})
    tracker.observe(6, {"val/unleaked_accuracy": 0.5})  # streak resets
    for epoch in (7, 8, 9):
        tracker.observe(epoch, {"val/unleaked_accuracy": 1.0})
    assert tracker.summary() == {"grokked": True, "censored": False, "epochs_to_grok": 7}


def test_grok_tracker_censors_and_ignores_nan():
    tracker = stream_runs.GrokTracker(
        metric_key="val/unleaked_accuracy", threshold=0.99, patience=2
    )
    tracker.observe(0, {"val/unleaked_accuracy": float("nan")})
    tracker.observe(1, {})
    assert tracker.summary() == {"grokked": False, "censored": True, "epochs_to_grok": None}


# ---------------------------------------------------------------------------
# streaming behaviour
# ---------------------------------------------------------------------------


def test_throttle_streams_every_nth_row_plus_latest(tmp_path):
    run_dir = _make_run_dir(tmp_path, "run-a")
    (run_dir / "run.log").write_text("\n".join(_epoch_line(epoch) for epoch in range(130)) + "\n")
    sink = FakeSink()
    sidecar = _sidecar(tmp_path, sink, log_every=50)
    sidecar.poll_once()
    epochs = [payload["epoch"] for payload in sink.logs["run-a"]]
    assert epochs == [0.0, 50.0, 100.0, 129.0]  # stride rows, then the frontier
    assert sink.logs["run-a"][0]["val/accuracy"] == 0.8
    assert sink.finished == []  # manifest still running


def test_completion_flushes_final_row_summary_and_finishes(tmp_path):
    run_dir = _make_run_dir(tmp_path, "run-a")
    rows = [_epoch_line(epoch, unleaked=1.0 if epoch >= 6 else 0.4) for epoch in range(10)]
    rows.append("completed run-a | {'val/accuracy': 0.97, 'val/unleaked_accuracy': 1.0}")
    (run_dir / "run.log").write_text("\n".join(rows) + "\n")
    _set_status(run_dir, "completed")

    sink = FakeSink()
    _sidecar(tmp_path, sink, log_every=4).poll_once()

    summary = sink.summaries["run-a"]
    assert summary["grokked"] is True and summary["censored"] is False
    assert summary["epochs_to_grok"] == 6  # patience 3, streak starts at epoch 6
    assert summary["val/accuracy"] == 0.97
    assert summary["stream/complete"] is True
    assert sink.finished == ["run-a"]
    epochs = [payload["epoch"] for payload in sink.logs["run-a"]]
    assert epochs == [0.0, 4.0, 8.0, 9.0]


def test_late_appearing_run_log_streams_on_completion(tmp_path):
    """A run.log can appear long after discovery (an ensemble seed before its
    first ``snapshot.history_flush_epochs`` flush, or a whole run with the
    flush disabled): the sidecar must handle a run directory that stays
    log-less for many polls and then produces the whole history at once."""
    run_dir = _make_run_dir(tmp_path, "run-a")
    sink = FakeSink()
    sidecar = _sidecar(tmp_path, sink, log_every=5)
    sidecar.poll_once()
    assert len(sink.init_specs) == 1  # the live run opens at discovery
    assert "run-a" not in sink.logs

    (run_dir / "run.log").write_text("\n".join(_epoch_line(e) for e in range(11)) + "\n")
    _set_status(run_dir, "completed")
    sidecar.poll_once()
    epochs = [payload["epoch"] for payload in sink.logs["run-a"]]
    assert epochs == [0.0, 5.0, 10.0]
    assert sink.summaries["run-a"]["censored"] is True
    assert sink.finished == ["run-a"]


def test_wandb_errors_back_off_without_losing_rows(tmp_path):
    run_dir = _make_run_dir(tmp_path, "run-a")
    (run_dir / "run.log").write_text("\n".join(_epoch_line(e) for e in range(3)) + "\n")
    sink = FakeSink()
    sink.fail_next_logs = 1
    clock = [0.0]
    sidecar = _sidecar(tmp_path, sink, log_every=1, now=lambda: clock[0])

    sidecar.poll_once()
    assert "run-a" not in sink.logs  # first send failed

    clock[0] = 1.0  # still inside the backoff window
    sidecar.poll_once()
    assert "run-a" not in sink.logs

    clock[0] = 10.0  # past the first 5s backoff
    sidecar.poll_once()
    epochs = [payload["epoch"] for payload in sink.logs["run-a"]]
    assert epochs == [0.0, 1.0, 2.0]  # nothing was lost


def test_restart_resumes_without_duplicating_epochs(tmp_path):
    run_dir = _make_run_dir(tmp_path, "run-a")
    (run_dir / "run.log").write_text("\n".join(_epoch_line(epoch) for epoch in range(101)) + "\n")
    sink = FakeSink()
    sink.resume_epochs["run-a"] = 50  # a previous sidecar streamed up to epoch 50
    _sidecar(tmp_path, sink, log_every=25).poll_once()
    epochs = [payload["epoch"] for payload in sink.logs["run-a"]]
    assert epochs == [75.0, 100.0]  # 0/25/50 already on W&B


def test_already_closed_run_is_skipped_fast(tmp_path):
    run_dir = _make_run_dir(tmp_path, "run-a", status="completed")
    (run_dir / "run.log").write_text(_epoch_line(0) + "\n")
    sink = FakeSink()
    sink.complete_ids.add("run-a")
    _sidecar(tmp_path, sink, log_every=1).poll_once()
    assert "run-a" not in sink.logs
    assert sink.finished == ["run-a"]


def test_malformed_lines_are_skipped(tmp_path, capsys):
    run_dir = _make_run_dir(tmp_path, "run-a")
    (run_dir / "run.log").write_text(
        _epoch_line(0) + "\n" + "epoch 1 | {'val/accuracy': }\n" + _epoch_line(2) + "\n"
    )
    sink = FakeSink()
    _sidecar(tmp_path, sink, log_every=1).poll_once()
    epochs = [payload["epoch"] for payload in sink.logs["run-a"]]
    assert epochs == [0.0, 2.0]
    assert "malformed" in capsys.readouterr().err


def test_discovery_waits_for_resolved_config(tmp_path):
    run_dir = tmp_path / "run-a"
    run_dir.mkdir()
    (run_dir / "manifest.yaml").write_text(yaml.safe_dump({"run_id": "run-a"}))
    sink = FakeSink()
    sidecar = _sidecar(tmp_path, sink)
    sidecar.poll_once()
    assert sidecar.streams == {}  # not adopted until the config lands


# ---------------------------------------------------------------------------
# snapshots and progress
# ---------------------------------------------------------------------------


def test_snapshot_epochs_parses_all_families(tmp_path):
    ckpt = tmp_path / "checkpoints"
    ckpt.mkdir()
    for name in ("step_128.pt", "generalized_step_400.pt", "final_epoch_501.pt", "final.pt"):
        (ckpt / name).touch()
    epochs = stream_runs.snapshot_epochs(ckpt)
    assert [epoch for epoch, _ in epochs] == [128, 400, 501]
    assert stream_runs.newest_snapshot_epoch(tmp_path) == 501
    assert stream_runs.snapshot_epochs(tmp_path / "absent") == []


def test_progress_rows_follow_new_snapshots(tmp_path):
    run_dir = _make_run_dir(tmp_path, "run-a")
    ckpt = run_dir / "checkpoints"
    ckpt.mkdir()
    (ckpt / "step_128.pt").touch()
    sink = FakeSink()
    sidecar = _sidecar(tmp_path, sink)

    sidecar.poll_once()
    assert sink.logs["run-a"] == [{"progress/snapshot_epoch": 128.0}]

    sidecar.poll_once()  # unchanged: no duplicate
    assert len(sink.logs["run-a"]) == 1

    (ckpt / "step_256.pt").touch()
    sidecar.poll_once()
    assert sink.logs["run-a"][-1] == {"progress/snapshot_epoch": 256.0}


# ---------------------------------------------------------------------------
# live occupancy
# ---------------------------------------------------------------------------


def test_occupancy_round_robin_is_bounded(tmp_path, monkeypatch):
    _make_run_dir(tmp_path, "run-a")
    _make_run_dir(tmp_path, "run-b")
    measured: list[str] = []

    def fake_measure(run_dir: Path, **_: Any) -> tuple[int, dict[str, float]]:
        measured.append(run_dir.name)
        return 1, {"occupancy/left/nontrivial/tv_to_null": 0.1}

    monkeypatch.setattr(stream_runs, "measure_snapshot_occupancy", fake_measure)
    clock = [0.0]
    sink = FakeSink()
    sidecar = _sidecar(tmp_path, sink, occupancy_every_s=10.0, now=lambda: clock[0])

    for tick in range(3):
        clock[0] = 100.0 * (tick + 1)
        sidecar.poll_once()
    assert measured == ["run-a", "run-b", "run-a"]  # one run per tick, round-robin
    # The repeat measurement of run-a found the same snapshot epoch: no new row.
    occupancy_rows = [
        payload
        for payload in sink.logs["run-a"]
        if "occupancy/left/nontrivial/tv_to_null" in payload
    ]
    assert len(occupancy_rows) == 1
    assert occupancy_rows[0]["epoch"] == 1.0


def test_occupancy_tick_respects_interval(tmp_path, monkeypatch):
    _make_run_dir(tmp_path, "run-a")
    calls: list[str] = []
    monkeypatch.setattr(
        stream_runs,
        "measure_snapshot_occupancy",
        lambda run_dir, **_: calls.append(run_dir.name) or (1, {}),
    )
    clock = [100.0]
    sidecar = _sidecar(tmp_path, FakeSink(), occupancy_every_s=60.0, now=lambda: clock[0])
    sidecar.poll_once()
    clock[0] = 130.0
    sidecar.poll_once()  # inside the interval
    assert calls == ["run-a"]
    clock[0] = 200.0
    sidecar.poll_once()
    assert calls == ["run-a", "run-a"]


@pytest.fixture(scope="module")
def real_run(tmp_path_factory) -> Path:
    """A real short single-seed training run on the fixture corpus, giving a
    genuine run.log (prefixed dialect), manifest, and trajectory snapshots."""
    root = tmp_path_factory.mktemp("stream-runs")
    config = ProjectConfig(
        device="cpu",
        seed=0,
        data={"group": "C4", "train_frac": 0.5, "split_seed": 0},
        model={"d_model": 16, "d_mlp": 32, "n_heads": 1},
        optim={"epochs": 6, "log_every": 1, "print_every": 1},
        snapshot={
            "enabled": True,
            "interval": 2,
            "log_dense_until": 2,
            "event_based": False,
            "final_window_epochs": 2,
        },
        logging=LoggingConfig(mode="disabled"),
        experiment=ExperimentConfig(name="stream-live"),
    )
    experiment = GroupGeneralizationExperiment(config, runs_root=root)
    experiment.execute()
    return experiment.run_dir


def test_live_occupancy_on_a_real_snapshot(real_run):
    root = real_run.parent
    _set_status(real_run, "running")  # keep the stream live so occupancy runs
    clock = [1000.0]
    sink = FakeSink()
    sidecar = _sidecar(
        root,
        sink,
        log_every=2,
        occupancy_every_s=1.0,
        occupancy_threshold=0.0,
        now=lambda: clock[0],
    )
    sidecar.poll_once()
    handle = real_run.name
    occupancy_rows = [
        payload
        for payload in sink.logs[handle]
        if "occupancy/left/nontrivial/tv_to_null" in payload
    ]
    assert len(occupancy_rows) == 1
    row = occupancy_rows[0]
    assert row["epoch"] == row["occupancy/snapshot_epoch"]
    for argument in ("left", "right"):
        tv = row[f"occupancy/{argument}/nontrivial/tv_to_null"]
        ratio = row[f"occupancy/{argument}/nontrivial/tv_over_floor"]
        share = row[f"occupancy/{argument}/trivial_share"]
        assert 0.0 <= tv <= 1.0
        assert ratio > 0.0
        assert 0.0 <= share <= 1.0

    # The real run.log (prefixed dialect) streamed alongside.
    streamed_epochs = [p["epoch"] for p in sink.logs[handle] if "val/accuracy" in p]
    assert streamed_epochs and streamed_epochs[0] == 0.0


def test_real_run_completion_summary(real_run):
    root = real_run.parent
    _set_status(real_run, "completed")
    sink = FakeSink()
    _sidecar(root, sink, log_every=2).poll_once()
    handle = real_run.name
    summary = sink.summaries[handle]
    assert summary["censored"] is True  # a 6-epoch run never sustains 0.99
    assert summary["stream/complete"] is True
    assert sink.finished == [handle]


# ---------------------------------------------------------------------------
# CLI gate
# ---------------------------------------------------------------------------


def test_main_refuses_to_run_offline():
    # conftest.py forces WANDB_MODE=disabled process-wide -- exactly the
    # environment in which the sidecar must refuse to start.
    assert stream_runs.main(["--once"]) == 2


def test_main_rejects_bad_log_every(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "online")
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    monkeypatch.setattr(stream_runs, "WandbSink", lambda *a, **k: FakeSink())
    assert stream_runs.main(["--once", "--log-every", "0"]) == 2


def test_main_once_runs_with_injected_sink(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "online")
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    sink = FakeSink()
    monkeypatch.setattr(stream_runs, "WandbSink", lambda *a, **k: sink)
    _make_run_dir(tmp_path, "run-a")
    (tmp_path / "run-a" / "run.log").write_text(_epoch_line(0) + "\n")
    assert stream_runs.main(["--once", "--runs-root", str(tmp_path)]) == 0
    assert [payload["epoch"] for payload in sink.logs["run-a"]] == [0.0]
