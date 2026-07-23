"""Focused tests for `scripts/backfill_runs.py` (the bounded-concurrency W&B
backfill sweeper).

Offline and network-free: the W&B boundary is the injectable ``WandbSink``
seam (from ``stream_runs.py``), replaced here by recording fakes; run
directories are synthesised on disk. Loaded by file path (mirrors
``tests/test_stream_runs.py`` and ``test_sync_runs.py``), since ``scripts/``
is not a package.

The core regression guard is ``test_bounded_concurrency_never_exceeded``: the
whole point of this script is that it must never repeat ``stream_runs.py``'s
saturation failure (hundreds of W&B runs held open at once), so a fake sink
tracks the live-open-run count under a lock and the test asserts it never
exceeds ``max_concurrent`` while a bounded pool works through many runs.
"""

from __future__ import annotations

import importlib.util
import sys
import threading
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"


def _load_script(name: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


stream_runs = _load_script("stream_runs")
backfill_runs = _load_script("backfill_runs")


class FakeSink:
    """Records every sink call; thread-safe so it can back a concurrent pool.
    Tracks the live-open-run count so tests can assert the bounded-
    concurrency invariant directly."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.init_specs: list[Any] = []
        self.logs: dict[str, list[dict[str, float]]] = {}
        self.summaries: dict[str, dict[str, Any]] = {}
        self.finished: list[str] = []
        self.resume_epochs: dict[str, int] = {}
        self.open_count = 0
        self.max_open_seen = 0
        self.open_delay_s = 0.0
        self.fail_next_logs_for: set[str] = set()

    def init_run(self, spec: Any) -> str:
        with self._lock:
            self.init_specs.append(spec)
            self.open_count += 1
            self.max_open_seen = max(self.max_open_seen, self.open_count)
        if self.open_delay_s:
            time.sleep(self.open_delay_s)
        return str(spec.run_id)

    def last_epoch(self, handle: str) -> int:
        return self.resume_epochs.get(handle, -1)

    def log(self, handle: str, payload: dict[str, float]) -> None:
        if handle in self.fail_next_logs_for:
            self.fail_next_logs_for.discard(handle)
            raise RuntimeError("simulated W&B outage")
        with self._lock:
            self.logs.setdefault(handle, []).append(payload)

    def set_summary(self, handle: str, mapping: dict[str, Any]) -> None:
        with self._lock:
            self.summaries.setdefault(handle, {}).update(mapping)

    def finish(self, handle: str) -> None:
        with self._lock:
            self.finished.append(handle)
            self.open_count -= 1


def _make_run_dir(
    root: Path,
    run_id: str,
    *,
    status: str = "running",
    completed_at: str | None = None,
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
                "completed_at": completed_at,
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


def _finalise(run_dir: Path, status: str = "completed") -> None:
    manifest = yaml.safe_load((run_dir / "manifest.yaml").read_text())
    manifest["status"] = status
    manifest["completed_at"] = "2026-07-20T00:00:00+00:00"
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


def _write_log(run_dir: Path, epochs: range | list[int], **kwargs: Any) -> None:
    (run_dir / "run.log").write_text("\n".join(_epoch_line(e, **kwargs) for e in epochs) + "\n")


def _backfiller(root: Path, sink: Any, **kwargs: Any) -> Any:
    return backfill_runs.Backfiller(root, sink, **kwargs)


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------


def test_discover_requires_manifest_config_and_log(tmp_path):
    complete = _make_run_dir(tmp_path, "run-complete")
    (complete / "run.log").write_text(_epoch_line(0) + "\n")

    no_log = _make_run_dir(tmp_path, "run-no-log")

    no_config = tmp_path / "run-no-config"
    no_config.mkdir()
    (no_config / "manifest.yaml").write_text(yaml.safe_dump({"run_id": "run-no-config"}))
    (no_config / "run.log").write_text(_epoch_line(0) + "\n")

    found = backfill_runs.discover_run_dirs(tmp_path)
    assert [p.name for p in found] == ["run-complete"]
    assert no_log.is_dir() and no_config.is_dir()  # sanity: they exist, just not discovered


def test_pruned_run_is_still_discovered_and_processed(tmp_path):
    """A run pruned by ship_runs.py keeps run.log/manifest/resolved_config
    and loses only checkpoints/ -- discovery must not gate on it."""
    run_dir = _make_run_dir(tmp_path, "run-pruned", status="completed")
    _finalise(run_dir)
    _write_log(run_dir, range(5))
    assert not (run_dir / "checkpoints").exists()

    sink = FakeSink()
    outcomes = _backfiller(tmp_path, sink, log_every=2).poll_once()

    assert [o.run_id for o in outcomes] == ["run-pruned"]
    assert outcomes[0].done is True
    assert sink.finished == ["run-pruned"]
    epochs = [p["epoch"] for p in sink.logs["run-pruned"] if "epoch" in p]
    assert epochs == [0.0, 2.0, 4.0]


# ---------------------------------------------------------------------------
# ledger idempotency: only NEW epochs go out on a second sweep
# ---------------------------------------------------------------------------


def test_second_sweep_appends_only_new_epochs(tmp_path):
    run_dir = _make_run_dir(tmp_path, "run-a")
    _write_log(run_dir, range(51))  # 0..50
    sink = FakeSink()
    backfiller = _backfiller(tmp_path, sink, log_every=25)

    backfiller.poll_once()
    first_epochs = [p["epoch"] for p in sink.logs["run-a"]]
    assert first_epochs == [0.0, 25.0, 50.0]

    # More epochs arrive; a second sweep must send only the new ones.
    _write_log(run_dir, range(76))  # 0..75
    backfiller.poll_once()
    all_epochs = [p["epoch"] for p in sink.logs["run-a"]]
    assert all_epochs == [0.0, 25.0, 50.0, 75.0]  # no duplicate of 0/25/50


def test_terminal_run_is_marked_done_and_skipped_thereafter(tmp_path):
    run_dir = _make_run_dir(tmp_path, "run-a")
    _write_log(run_dir, range(10))
    _finalise(run_dir)
    sink = FakeSink()
    backfiller = _backfiller(tmp_path, sink, log_every=5)

    outcomes = backfiller.poll_once()
    assert outcomes[0].done is True
    assert sink.finished == ["run-a"]
    entry = backfill_runs.ledger_entry(backfiller.ledger, "run-a")
    assert entry["done"] is True

    # A second sweep must not even open W&B again for this run.
    backfiller.poll_once()
    assert sink.init_specs == [sink.init_specs[0]]  # unchanged: no second init_run
    assert sink.finished == ["run-a"]


def test_ledger_persists_and_reloads_across_backfiller_instances(tmp_path):
    run_dir = _make_run_dir(tmp_path, "run-a")
    _write_log(run_dir, range(10))
    sink = FakeSink()
    _backfiller(tmp_path, sink, log_every=5).poll_once()

    ledger_path = backfill_runs.default_ledger_path(tmp_path)
    assert ledger_path.is_file()
    reloaded = backfill_runs.load_ledger(ledger_path)
    assert reloaded["runs"]["run-a"]["last_logged_epoch"] == 9

    # A fresh Backfiller over the same runs-root picks up the ledger state.
    sink2 = FakeSink()
    second = _backfiller(tmp_path, sink2, log_every=5)
    assert backfill_runs.ledger_entry(second.ledger, "run-a")["last_logged_epoch"] == 9


# ---------------------------------------------------------------------------
# per-run error handling: a bad run must not stop the sweep, and must always
# still close whatever it opened
# ---------------------------------------------------------------------------


def test_unreadable_manifest_is_skipped_and_retried(tmp_path):
    run_dir = _make_run_dir(tmp_path, "run-a")
    _write_log(run_dir, range(1))
    (run_dir / "manifest.yaml").write_text("{ not: valid: yaml [")
    sink = FakeSink()
    outcomes = _backfiller(tmp_path, sink, log_every=1).poll_once()
    assert outcomes[0].error is not None
    assert outcomes[0].last_logged_epoch == -1
    assert sink.init_specs == []  # never got far enough to open a W&B run


def test_build_run_spec_failure_is_skipped_and_retried(tmp_path):
    run_dir = _make_run_dir(tmp_path, "run-a")
    _write_log(run_dir, range(1))
    (run_dir / "resolved_config.yaml").write_text("d_model: [1, 2")  # invalid YAML
    sink = FakeSink()
    outcomes = _backfiller(tmp_path, sink, log_every=1).poll_once()
    assert outcomes[0].error is not None
    assert sink.init_specs == []  # never got far enough to open a W&B run


def test_init_run_failure_is_skipped_and_retried(tmp_path):
    run_dir = _make_run_dir(tmp_path, "run-a")
    _write_log(run_dir, range(1))

    class _FailingInitSink(FakeSink):
        def init_run(self, spec: Any) -> str:
            raise RuntimeError("simulated connection failure")

    sink = _FailingInitSink()
    outcomes = _backfiller(tmp_path, sink, log_every=1).poll_once()
    assert outcomes[0].error is not None
    assert outcomes[0].last_logged_epoch == -1
    assert sink.finished == []  # no handle was ever obtained, so nothing to close


# ---------------------------------------------------------------------------
# in-progress runs keep getting revisited
# ---------------------------------------------------------------------------


def test_in_progress_run_is_revisited_every_sweep(tmp_path):
    run_dir = _make_run_dir(tmp_path, "run-a", status="running")
    _write_log(run_dir, range(1))
    sink = FakeSink()
    backfiller = _backfiller(tmp_path, sink, log_every=1)

    backfiller.poll_once()
    assert not backfill_runs.ledger_entry(backfiller.ledger, "run-a")["done"]

    _write_log(run_dir, range(3))
    backfiller.poll_once()
    epochs = [p["epoch"] for p in sink.logs["run-a"]]
    assert epochs == [0.0, 1.0, 2.0]
    assert sink.finished == ["run-a", "run-a"]  # closed after every sweep's visit


# ---------------------------------------------------------------------------
# reconciling a stalled sidecar's half-open runs
# ---------------------------------------------------------------------------


def test_resume_reconciles_a_half_open_run_from_the_old_sidecar(tmp_path):
    """A run the stalled sidecar held open (never finish()-ed) already has a
    server-side summary epoch beyond what the local ledger knows about --
    resume must not resend those epochs, and must still close the run out."""
    run_dir = _make_run_dir(tmp_path, "run-a")
    _write_log(run_dir, range(101))  # 0..100
    _finalise(run_dir)
    sink = FakeSink()
    sink.resume_epochs["run-a"] = 60  # the old stuck sidecar streamed up to epoch 60

    outcomes = _backfiller(tmp_path, sink, log_every=25).poll_once()

    epochs = [p["epoch"] for p in sink.logs["run-a"] if "epoch" in p]
    assert epochs == [75.0, 100.0]  # 0/25/50/60 were already on W&B
    assert outcomes[0].done is True
    assert sink.finished == ["run-a"]
    assert sink.summaries["run-a"]["stream/complete"] is True


def test_wandb_error_does_not_prevent_finish(tmp_path):
    """Even when logging raises mid-flush, finish() must still be called --
    the anti-saturation invariant holds under errors, not just the happy
    path."""
    run_dir = _make_run_dir(tmp_path, "run-a")
    _write_log(run_dir, range(3))
    sink = FakeSink()
    sink.fail_next_logs_for.add("run-a")

    outcomes = _backfiller(tmp_path, sink, log_every=1).poll_once()

    assert sink.finished == ["run-a"]  # closed despite the log failure
    assert outcomes[0].error is not None
    # Retried next sweep: ledger was not advanced past the failed row.
    assert outcomes[0].last_logged_epoch == -1


# ---------------------------------------------------------------------------
# the core regression guard: bounded concurrency
# ---------------------------------------------------------------------------


def test_bounded_concurrency_never_exceeded(tmp_path):
    """Many runs, a small max_concurrent, and an artificial delay in
    init_run so pool workers genuinely overlap. The number of simultaneously
    open W&B runs must never exceed max_concurrent -- this is the regression
    scripts/stream_runs.py hit at ~350 concurrently open runs."""
    n_runs = 20
    for i in range(n_runs):
        run_dir = _make_run_dir(tmp_path, f"run-{i:02d}")
        _write_log(run_dir, range(3))

    sink = FakeSink()
    sink.open_delay_s = 0.02
    max_concurrent = 3

    outcomes = _backfiller(tmp_path, sink, log_every=1, max_concurrent=max_concurrent).poll_once()

    assert len(outcomes) == n_runs
    assert sink.max_open_seen <= max_concurrent
    assert sink.max_open_seen > 1  # sanity: the pool did overlap work, not run serially by luck
    assert sink.open_count == 0  # every run was finished by the end of the sweep
    assert len(sink.finished) == n_runs


def test_max_concurrent_of_one_is_fully_sequential(tmp_path):
    for i in range(5):
        run_dir = _make_run_dir(tmp_path, f"run-{i:02d}")
        _write_log(run_dir, range(2))
    sink = FakeSink()
    sink.open_delay_s = 0.01
    outcomes = _backfiller(tmp_path, sink, log_every=1, max_concurrent=1).poll_once()
    assert len(outcomes) == 5
    assert sink.max_open_seen == 1


# ---------------------------------------------------------------------------
# --once and the CLI
# ---------------------------------------------------------------------------


def test_run_forever_once_does_a_single_sweep(tmp_path, monkeypatch):
    run_dir = _make_run_dir(tmp_path, "run-a")
    _write_log(run_dir, range(1))
    sink = FakeSink()
    backfiller = _backfiller(tmp_path, sink, log_every=1)

    calls: list[None] = []
    original = backfiller.poll_once

    def _counted() -> list[Any]:
        calls.append(None)
        return original()

    monkeypatch.setattr(backfiller, "poll_once", _counted)
    backfiller.run_forever(interval=1000.0, once=True)
    assert len(calls) == 1


def test_run_forever_stops_at_complete_sentinel(tmp_path):
    run_dir = _make_run_dir(tmp_path, "run-a")
    _write_log(run_dir, range(1))
    sink = FakeSink()
    backfiller = _backfiller(tmp_path, sink, log_every=1)
    sentinel = tmp_path / "CAMPAIGN_COMPLETE"
    sentinel.write_text("done\n")

    sleeps: list[float] = []
    backfiller.run_forever(interval=5.0, complete_sentinel=sentinel, sleep=sleeps.append)
    assert sleeps == []  # never slept: the sentinel was already present


def test_main_refuses_to_run_offline():
    # conftest.py forces WANDB_MODE=disabled process-wide.
    assert backfill_runs.main(["--once"]) == 2


def test_main_rejects_bad_log_every(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "online")
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    monkeypatch.setattr(backfill_runs.stream_runs, "WandbSink", lambda *a, **k: FakeSink())
    assert backfill_runs.main(["--once", "--log-every", "0"]) == 2


def test_main_rejects_bad_max_concurrent(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "online")
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    monkeypatch.setattr(backfill_runs.stream_runs, "WandbSink", lambda *a, **k: FakeSink())
    assert backfill_runs.main(["--once", "--max-concurrent", "0"]) == 2


def test_main_once_runs_with_injected_sink(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "online")
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    sink = FakeSink()
    monkeypatch.setattr(backfill_runs.stream_runs, "WandbSink", lambda *a, **k: sink)
    run_dir = _make_run_dir(tmp_path, "run-a")
    _write_log(run_dir, range(1))
    assert backfill_runs.main(["--once", "--runs-root", str(tmp_path)]) == 0
    assert [p["epoch"] for p in sink.logs["run-a"] if "epoch" in p] == [0.0]
