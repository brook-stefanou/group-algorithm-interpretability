"""Focused tests for `scripts/ship_runs.py` (the curated-snapshot S3 shipper).

Offline and network-free: boto3 is never installed in this environment, and
`S3Sink` only imports it lazily inside its own constructor, so the module
itself must import fine without it (asserted directly below). The S3 boundary
used everywhere else is a hand-rolled in-memory fake (`FakeS3Sink`) with the
same `put_file`/`head` surface `S3Sink` exposes. Run directories are
synthesised on disk (mirrors `tests/test_stream_runs.py`); the script is an
executable entry point, not part of the installed package, so it is loaded by
file path.

The dominant concern under test is the verify-before-prune invariant: a
previous campaign lost 50 runs to a prune that ran ahead of verification, so
every test that can plausibly threaten that ordering asserts the local
`checkpoints/` directory survives byte-for-byte until every curated object is
confirmed present in S3 with a matching size.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "ship_runs.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ship_runs", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["ship_runs"] = module
    spec.loader.exec_module(module)
    return module


ship_runs = _load_script()

EPOCHS = 30
SNAPSHOT_STEP = 2
NEVER = EPOCHS + 100  # an onset value past the log range: "never reached"


# ---------------------------------------------------------------------------
# fixtures / fakes
# ---------------------------------------------------------------------------


class FakeS3Sink:
    """Records every call; upload/verify failures are injectable per key."""

    def __init__(self) -> None:
        self.objects: dict[str, int] = {}
        self.contents: dict[str, bytes] = {}
        self.put_calls: list[str] = []
        self.head_calls: list[str] = []
        self.fail_put_count: dict[str, int] = {}
        self.fail_head_keys: set[str] = set()
        self.corrupt_head_keys: set[str] = set()

    def put_file(self, key: str, local_path: Path) -> None:
        self.put_calls.append(key)
        remaining = self.fail_put_count.get(key, 0)
        if remaining > 0:
            self.fail_put_count[key] = remaining - 1
            raise RuntimeError("simulated transient upload failure")
        self.objects[key] = local_path.stat().st_size
        self.contents[key] = local_path.read_bytes()

    def head(self, key: str) -> int | None:
        self.head_calls.append(key)
        if key in self.fail_head_keys:
            return None
        if key in self.corrupt_head_keys:
            return -1
        return self.objects.get(key)


def _epoch_metrics(epoch: int, leaked_onset: int, unleaked_onset: int) -> dict[str, float]:
    leaked = 0.995 if epoch >= leaked_onset else 0.3 + 0.01 * epoch
    unleaked = 0.995 if epoch >= unleaked_onset else 0.2 + 0.01 * epoch
    return {
        "train/loss": 0.01,
        "train/accuracy": leaked,
        "val/loss": 0.02,
        "val/accuracy": leaked,
        "val/unleaked_accuracy": unleaked,
        "val/weight_norm": 12.0,
    }


def _write_checkpoint(path: Path, tag: str) -> None:
    path.write_bytes((f"checkpoint::{tag}::").encode() * 8)


def _build_run(
    root: Path,
    run_id: str,
    *,
    status: str = "completed",
    finalized: bool = True,
    leaked_onset: int = 10,
    unleaked_onset: int = 20,
    epochs: int = EPOCHS,
    snapshot_step: int = SNAPSHOT_STEP,
    include_final: bool = True,
    write_log: bool = True,
) -> Path:
    run_dir = root / run_id
    (run_dir / "checkpoints").mkdir(parents=True)
    manifest = {
        "run_id": run_id,
        "status": status,
        "completed_at": "2026-07-20T00:00:00+00:00" if finalized else None,
    }
    (run_dir / "manifest.yaml").write_text(yaml.safe_dump(manifest))
    # No snapshot.final_window_epochs -> select_checkpoint takes the
    # final_gate rule (matches an "old-style" resolved config).
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump({"snapshot": {}}))

    if write_log:
        lines = [
            f"epoch {epoch} | {_epoch_metrics(epoch, leaked_onset, unleaked_onset)}"
            for epoch in range(epochs + 1)
        ]
        (run_dir / "run.log").write_text("\n".join(lines) + "\n")

    for epoch in range(0, epochs + 1, snapshot_step):
        _write_checkpoint(run_dir / "checkpoints" / f"step_{epoch}.pt", f"step-{epoch}")
    if include_final:
        _write_checkpoint(run_dir / "checkpoints" / "final.pt", "final")

    return run_dir


def _key(run_id: str, filename: str, prefix: str = ship_runs.DEFAULT_KEY_PREFIX) -> str:
    return f"{prefix}/{run_id}/{filename}"


# ---------------------------------------------------------------------------
# module import / laziness
# ---------------------------------------------------------------------------


def test_module_imports_without_boto3_installed():
    assert importlib.util.find_spec("boto3") is None, "test assumes boto3 is not installed"
    assert ship_runs is not None


def test_s3_sink_construction_is_lazy_about_boto3():
    with pytest.raises(ModuleNotFoundError):
        ship_runs.S3Sink("bucket", endpoint_url=None, region_name=None)


# ---------------------------------------------------------------------------
# small pure helpers
# ---------------------------------------------------------------------------


def test_snapshot_epochs_orders_ascending_and_ignores_final(tmp_path):
    ckpt = tmp_path / "checkpoints"
    ckpt.mkdir()
    for name in ("step_4.pt", "step_0.pt", "generalized_step_2.pt", "final.pt", "not_a_ckpt.txt"):
        (ckpt / name).write_bytes(b"x")
    assert [e for e, _ in ship_runs.snapshot_epochs(ckpt)] == [0, 2, 4]


def test_snapshot_epochs_missing_dir_is_empty(tmp_path):
    assert ship_runs.snapshot_epochs(tmp_path / "nope") == []


@pytest.mark.parametrize(
    "n,expected_indices",
    [(0, []), (1, [0]), (2, [0, 1]), (3, [1, 2]), (4, [1, 2]), (7, [2, 4])],
)
def test_pick_intermediate_picks_one_or_two_evenly_spaced(n, expected_indices):
    candidates = [(i, Path(f"p{i}")) for i in range(n)]
    picked = ship_runs._pick_intermediate(candidates)
    assert picked == [candidates[i] for i in expected_indices]
    assert len(picked) <= 2


def test_find_unleaked_metric_key_from_first_row():
    rows = [
        ship_runs.EpochRow(epoch=0, metrics={"val/accuracy": 0.5, "val/unleaked_accuracy": 0.1})
    ]
    assert ship_runs.find_unleaked_metric_key(rows) == "val/unleaked_accuracy"


def test_find_unleaked_metric_key_absent():
    rows = [ship_runs.EpochRow(epoch=0, metrics={"val/accuracy": 0.5})]
    assert ship_runs.find_unleaked_metric_key(rows) is None
    assert ship_runs.find_unleaked_metric_key([]) is None


# ---------------------------------------------------------------------------
# curated selection
# ---------------------------------------------------------------------------


def test_curated_selection_full_run_picks_every_category(tmp_path):
    run_dir = _build_run(tmp_path, "run-full", leaked_onset=10, unleaked_onset=20)
    selection = ship_runs.build_curated_selection(run_dir, manifest_status="completed")

    assert selection.censored is False
    assert selection.interrupted is False
    assert selection.leaked_metric_key == "val/accuracy"
    assert selection.unleaked_metric_key == "val/unleaked_accuracy"

    by_category: dict[str, list[Any]] = {}
    for entry in selection.entries:
        by_category.setdefault(entry.category, []).append(entry)

    assert by_category["first"][0].epoch == 0
    assert by_category["leaked_onset"][0].epoch == 10
    assert by_category["leaked_onset"][0].metric_value == pytest.approx(0.995)
    assert by_category["unleaked_onset"][0].epoch == 20
    assert [e.epoch for e in by_category["intermediate"]] == [14, 16]
    assert by_category["last"][0].epoch == EPOCHS
    assert by_category["last"][0].path.name == "final.pt"
    assert by_category["stable_end"][0].epoch == EPOCHS
    assert by_category["stable_end"][0].path.name == "final.pt"
    # A genuinely unsubstituted pick still carries the fields, just empty/None
    # -- never omitted (finding 1b: future ships write the full record).
    assert by_category["stable_end"][0].substitution is None
    assert by_category["stable_end"][0].rejected == []

    # final.pt is chosen by both "last" and "stable_end": dedup collapses it
    # to one physical upload even though two categories reference it.
    assert len(selection.entries) == 7
    assert len(selection.unique_files()) == 6


def test_curated_selection_stable_end_carries_substitution_and_rejected_trail(tmp_path):
    """Finding 1(b): a shipped ``categories.stable_end`` must carry
    ``substitution``/``rejected``, not just epoch/metric_value/filename --
    the trail a post-hoc reconstruction from ``selection.json`` alone would
    otherwise have to fabricate. Here the final epoch's leaked accuracy dips
    below the bar, so stable_end substitutes the nearest stable trajectory
    snapshot instead of final.pt."""
    run_dir = _build_run(tmp_path, "run-dip", leaked_onset=10, unleaked_onset=20)
    dipped_final = _epoch_metrics(EPOCHS, 10, 20)
    dipped_final["val/accuracy"] = 0.5
    lines = [f"epoch {epoch} | {_epoch_metrics(epoch, 10, 20)}" for epoch in range(EPOCHS)]
    lines.append(f"epoch {EPOCHS} | {dipped_final}")
    (run_dir / "run.log").write_text("\n".join(lines) + "\n")

    selection = ship_runs.build_curated_selection(run_dir, manifest_status="completed")
    stable_end = next(e for e in selection.entries if e.category == "stable_end")
    assert stable_end.path.name != "final.pt"
    assert stable_end.substitution == "trajectory"
    assert stable_end.rejected
    assert stable_end.rejected[0]["checkpoint"] == "final.pt"
    assert stable_end.rejected[0]["value"] == pytest.approx(0.5)

    entry_record = stable_end.to_record()
    assert entry_record["substitution"] == "trajectory"
    assert entry_record["rejected"] == stable_end.rejected

    full_record = selection.to_record()
    stable_end_record = full_record["categories"]["stable_end"]
    assert stable_end_record["substitution"] == "trajectory"
    assert stable_end_record["rejected"]
    # Round-trips through JSON exactly like it will when shipped.
    reparsed = json.loads(json.dumps(full_record))
    assert reparsed["categories"]["stable_end"]["substitution"] == "trajectory"
    # A non-stable_end category never carries substitution semantics.
    assert "substitution" not in full_record["categories"]["last"]


def test_curated_selection_censored_run_skips_onset_and_intermediate(tmp_path):
    run_dir = _build_run(tmp_path, "run-censored", leaked_onset=NEVER, unleaked_onset=NEVER)
    selection = ship_runs.build_curated_selection(run_dir, manifest_status="completed")

    assert selection.censored is True
    categories = {e.category for e in selection.entries}
    assert "leaked_onset" not in categories
    assert "unleaked_onset" not in categories
    assert "intermediate" not in categories
    assert "first" in categories
    assert "last" in categories
    # Never reaches the bar anywhere: stable_end selects nothing, and that is
    # recorded as an anomaly rather than silently omitted.
    assert "stable_end" not in categories
    assert any("stable_end" in a for a in selection.anomalies)

    record = selection.to_record()
    assert record["censored"] is True
    assert record["categories"]["leaked_onset"] is None
    assert record["categories"]["unleaked_onset"] is None
    assert record["categories"]["intermediate"] == []


def test_curated_selection_skips_intermediate_when_onsets_out_of_order(tmp_path):
    # Contrived: the unleaked metric clears the bar before the leaked one.
    run_dir = _build_run(tmp_path, "run-flipped", leaked_onset=20, unleaked_onset=10)
    selection = ship_runs.build_curated_selection(run_dir, manifest_status="completed")

    categories = {e.category: e for e in selection.entries if e.category != "intermediate"}
    assert categories["leaked_onset"].epoch == 20
    assert categories["unleaked_onset"].epoch == 10
    assert not any(e.category == "intermediate" for e in selection.entries)
    assert any("<=" in a for a in selection.anomalies)


def test_curated_selection_interrupted_run_falls_back_for_last(tmp_path):
    run_dir = _build_run(tmp_path, "run-interrupted", include_final=False)
    selection = ship_runs.build_curated_selection(run_dir, manifest_status="completed")

    assert selection.interrupted is True
    last_entries = [e for e in selection.entries if e.category == "last"]
    assert len(last_entries) == 1
    assert last_entries[0].path.name == f"step_{EPOCHS}.pt"
    assert any("final.pt missing" in a for a in selection.anomalies)


def test_curated_selection_interrupted_flag_follows_failed_status_even_with_final(tmp_path):
    run_dir = _build_run(tmp_path, "run-failed", status="failed", include_final=True)
    selection = ship_runs.build_curated_selection(run_dir, manifest_status="failed")
    assert selection.interrupted is True  # final.pt present, but status is not "completed"


def test_curated_selection_dedup_and_selection_json_round_trip(tmp_path):
    run_dir = _build_run(tmp_path, "run-json", leaked_onset=10, unleaked_onset=20)
    selection = ship_runs.build_curated_selection(run_dir, manifest_status="completed")
    record = selection.to_record()
    # Round-trips through JSON exactly like it will when shipped.
    reparsed = json.loads(json.dumps(record))
    assert reparsed["categories"]["leaked_onset"]["filename"] == "step_10.pt"
    assert reparsed["categories"]["unleaked_onset"]["filename"] == "step_20.pt"
    assert [c["filename"] for c in reparsed["categories"]["intermediate"]] == [
        "step_14.pt",
        "step_16.pt",
    ]


# ---------------------------------------------------------------------------
# completion detection
# ---------------------------------------------------------------------------


def test_is_shippable_rejects_running_manifest(tmp_path):
    run_dir = _build_run(tmp_path, "run-running", status="running", finalized=False)
    shippable, reason = ship_runs.is_shippable(run_dir)
    assert shippable is False
    assert "not finalised" in reason


def test_is_shippable_requires_final_checkpoint_unless_forced(tmp_path):
    run_dir = _build_run(tmp_path, "run-nofinal", include_final=False)
    shippable, reason = ship_runs.is_shippable(run_dir)
    assert shippable is False
    assert "final.pt" in reason

    shippable_forced, reason_forced = ship_runs.is_shippable(run_dir, force_interrupted=True)
    assert shippable_forced is True
    assert reason_forced is None


def test_is_shippable_accepts_finalised_run_with_final_checkpoint(tmp_path):
    run_dir = _build_run(tmp_path, "run-ok")
    shippable, reason = ship_runs.is_shippable(run_dir)
    assert shippable is True
    assert reason is None


# ---------------------------------------------------------------------------
# ship_run: uploads, verify-before-prune, retries
# ---------------------------------------------------------------------------


def test_ship_run_happy_path_uploads_verifies_and_prunes(tmp_path):
    run_dir = _build_run(tmp_path, "run-ship-ok")
    sink = FakeS3Sink()

    outcome = ship_runs.ship_run(
        run_dir,
        sink=sink,
        key_prefix="curated",
        threshold=0.99,
        manifest_status="completed",
        dry_run=False,
        warn=lambda _msg: None,
    )

    assert outcome.ok is True
    assert outcome.pruned is True
    assert not (run_dir / "checkpoints").exists()
    # Tiny provenance files survive locally.
    assert (run_dir / "manifest.yaml").is_file()
    assert (run_dir / "resolved_config.yaml").is_file()
    assert (run_dir / "run.log").is_file()

    uploaded_keys = {_key("run-ship-ok", name) for name in outcome.uploaded}
    assert uploaded_keys <= set(sink.objects)
    assert _key("run-ship-ok", "selection.json") in sink.objects
    assert _key("run-ship-ok", "run.log.gz") in sink.objects
    assert _key("run-ship-ok", "manifest.yaml") in sink.objects
    # Every uploaded key was verified with head_object before the prune.
    for key in uploaded_keys:
        assert key in sink.head_calls


def test_ship_run_blocks_prune_when_head_object_size_mismatches(tmp_path):
    run_dir = _build_run(tmp_path, "run-mismatch")
    before = sorted(p.name for p in (run_dir / "checkpoints").glob("*.pt"))
    sink = FakeS3Sink()
    corrupted_key = _key("run-mismatch", "final.pt")
    sink.corrupt_head_keys.add(corrupted_key)

    outcome = ship_runs.ship_run(
        run_dir,
        sink=sink,
        key_prefix="curated",
        threshold=0.99,
        manifest_status="completed",
        dry_run=False,
        warn=lambda _msg: None,
    )

    assert outcome.ok is False
    assert outcome.pruned is False
    assert "mismatch" in outcome.reason
    # HARD INVARIANT: nothing local is touched when verification fails.
    assert (run_dir / "checkpoints").is_dir()
    after = sorted(p.name for p in (run_dir / "checkpoints").glob("*.pt"))
    assert after == before


def test_ship_run_blocks_prune_when_head_object_reports_missing(tmp_path):
    run_dir = _build_run(tmp_path, "run-missing-object")
    before = sorted(p.name for p in (run_dir / "checkpoints").glob("*.pt"))
    sink = FakeS3Sink()
    sink.fail_head_keys.add(_key("run-missing-object", "manifest.yaml"))

    outcome = ship_runs.ship_run(
        run_dir,
        sink=sink,
        key_prefix="curated",
        threshold=0.99,
        manifest_status="completed",
        dry_run=False,
        warn=lambda _msg: None,
    )

    assert outcome.ok is False
    assert outcome.pruned is False
    assert (run_dir / "checkpoints").is_dir()
    after = sorted(p.name for p in (run_dir / "checkpoints").glob("*.pt"))
    assert after == before


def test_ship_run_blocks_prune_when_upload_fails_and_never_calls_head(tmp_path):
    run_dir = _build_run(tmp_path, "run-upload-fail")
    before = sorted(p.name for p in (run_dir / "checkpoints").glob("*.pt"))
    sink = FakeS3Sink()
    # Permanent failure (never recovers within the retry budget).
    sink.fail_put_count[_key("run-upload-fail", "final.pt")] = 999

    outcome = ship_runs.ship_run(
        run_dir,
        sink=sink,
        key_prefix="curated",
        threshold=0.99,
        manifest_status="completed",
        dry_run=False,
        warn=lambda _msg: None,
        retry_attempts=2,
        retry_base_delay=0.001,
    )

    assert outcome.ok is False
    assert "upload failed" in outcome.reason
    assert (run_dir / "checkpoints").is_dir()
    after = sorted(p.name for p in (run_dir / "checkpoints").glob("*.pt"))
    assert after == before
    # The verify phase never starts once an upload permanently fails.
    assert sink.head_calls == []


def test_ship_run_retries_transient_upload_failure_and_still_ships(tmp_path):
    run_dir = _build_run(tmp_path, "run-flaky")
    sink = FakeS3Sink()
    key = _key("run-flaky", "final.pt")
    sink.fail_put_count[key] = 2  # fails twice, succeeds on the 3rd attempt

    outcome = ship_runs.ship_run(
        run_dir,
        sink=sink,
        key_prefix="curated",
        threshold=0.99,
        manifest_status="completed",
        dry_run=False,
        warn=lambda _msg: None,
        retry_attempts=4,
        retry_base_delay=0.001,
    )

    assert outcome.ok is True
    assert outcome.pruned is True
    assert sink.put_calls.count(key) == 3


def test_ship_run_dry_run_uploads_and_prunes_nothing(tmp_path):
    run_dir = _build_run(tmp_path, "run-dry")
    before = sorted(p.name for p in (run_dir / "checkpoints").glob("*.pt"))
    sink = FakeS3Sink()

    outcome = ship_runs.ship_run(
        run_dir,
        sink=sink,
        key_prefix="curated",
        threshold=0.99,
        manifest_status="completed",
        dry_run=True,
        warn=lambda _msg: None,
    )

    assert outcome.ok is True
    assert outcome.pruned is False
    assert sink.put_calls == []
    assert sink.head_calls == []
    assert (run_dir / "checkpoints").is_dir()
    after = sorted(p.name for p in (run_dir / "checkpoints").glob("*.pt"))
    assert after == before


# ---------------------------------------------------------------------------
# ledger / Daemon
# ---------------------------------------------------------------------------


def test_ledger_round_trip(tmp_path):
    path = ship_runs.ledger_path_for(tmp_path)
    ledger = ship_runs.load_ledger(path)
    assert ledger == {"runs": {}}
    ship_runs.record_shipped(ledger, "run-a", pruned=True, files=["final.pt"], when="now")
    ship_runs.save_ledger(path, ledger)
    reloaded = ship_runs.load_ledger(path)
    assert ship_runs.is_ledgered(reloaded, "run-a")
    assert not ship_runs.is_ledgered(reloaded, "run-b")


def test_daemon_skips_running_runs_and_ships_only_finalised_ones(tmp_path):
    runs_root = tmp_path / "runs"
    _build_run(runs_root, "run-running", status="running", finalized=False)
    _build_run(runs_root, "run-done")
    sink = FakeS3Sink()
    daemon = ship_runs.Daemon(runs_root, sink, warn=lambda _msg: None)

    outcomes = daemon.poll_once()

    assert [o.run_id for o in outcomes] == ["run-done"]
    assert (runs_root / "run-running" / "checkpoints").is_dir()
    assert not (runs_root / "run-done" / "checkpoints").exists()


def test_daemon_ledger_idempotency_second_poll_ships_nothing_new(tmp_path):
    runs_root = tmp_path / "runs"
    _build_run(runs_root, "run-once")
    sink = FakeS3Sink()
    daemon = ship_runs.Daemon(runs_root, sink, warn=lambda _msg: None)

    first = daemon.poll_once()
    assert len(first) == 1
    assert first[0].ok is True
    puts_after_first = list(sink.put_calls)
    assert puts_after_first  # something was actually uploaded

    second = daemon.poll_once()
    assert second == []
    assert sink.put_calls == puts_after_first  # no new uploads on the re-poll

    # A fresh Daemon instance (simulating a process restart) reads the same
    # ledger off disk and also skips the already-shipped run.
    restarted = ship_runs.Daemon(runs_root, sink, warn=lambda _msg: None)
    assert restarted.poll_once() == []


def test_daemon_dry_run_never_writes_ledger(tmp_path):
    runs_root = tmp_path / "runs"
    _build_run(runs_root, "run-dry")
    sink = FakeS3Sink()
    daemon = ship_runs.Daemon(runs_root, sink, dry_run=True, warn=lambda _msg: None)

    outcomes = daemon.poll_once()
    assert outcomes[0].ok is True
    assert not ship_runs.is_ledgered(daemon.ledger, "run-dry")
    assert not daemon.ledger_path.is_file()
    assert (runs_root / "run-dry" / "checkpoints").is_dir()


def test_daemon_run_forever_once_processes_backlog_and_returns(tmp_path):
    runs_root = tmp_path / "runs"
    _build_run(runs_root, "run-a")
    _build_run(runs_root, "run-b")
    sink = FakeS3Sink()
    daemon = ship_runs.Daemon(runs_root, sink, warn=lambda _msg: None)

    def _boom(_seconds: float) -> None:
        raise AssertionError("run_forever(once=True) must never sleep")

    daemon.run_forever(1.0, once=True, sleep=_boom)

    assert ship_runs.is_ledgered(daemon.ledger, "run-a")
    assert ship_runs.is_ledgered(daemon.ledger, "run-b")


def test_daemon_run_forever_exits_after_final_sweep_on_sentinel(tmp_path):
    runs_root = tmp_path / "runs"
    _build_run(runs_root, "run-a")
    sink = FakeS3Sink()
    daemon = ship_runs.Daemon(runs_root, sink, warn=lambda _msg: None)

    poll_calls = 0
    original_poll_once = daemon.poll_once

    def counting_poll_once():
        nonlocal poll_calls
        poll_calls += 1
        return original_poll_once()

    daemon.poll_once = counting_poll_once  # type: ignore[method-assign]

    sentinel = tmp_path / "CAMPAIGN_COMPLETE"
    sentinel.write_text("done")

    def _boom(_seconds: float) -> None:
        raise AssertionError("must not sleep once the sentinel is already present")

    daemon.run_forever(1.0, once=False, complete_sentinel=sentinel, sleep=_boom)

    # One regular poll, then the sentinel triggers exactly one more "final
    # sweep" poll before returning -- never a sleep in between.
    assert poll_calls == 2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_dry_run_needs_no_bucket_and_never_touches_boto3(tmp_path, monkeypatch):
    monkeypatch.delenv("GAI_S3_BUCKET", raising=False)
    runs_root = tmp_path / "runs"
    _build_run(runs_root, "run-a")
    exit_code = ship_runs.main(["--dry-run", "--once", "--runs-root", str(runs_root)])
    assert exit_code == 0
    # Still present: dry-run touched nothing.
    assert (runs_root / "run-a" / "checkpoints").is_dir()


def test_cli_requires_bucket_when_not_dry_run(tmp_path, monkeypatch):
    monkeypatch.delenv("GAI_S3_BUCKET", raising=False)
    with pytest.raises(SystemExit) as excinfo:
        ship_runs.main(["--once", "--runs-root", str(tmp_path)])
    assert excinfo.value.code == 2


def test_cli_rejects_nonpositive_interval(tmp_path, monkeypatch):
    monkeypatch.setenv("GAI_S3_BUCKET", "some-bucket")
    with pytest.raises(SystemExit) as excinfo:
        ship_runs.main(["--once", "--interval", "0", "--runs-root", str(tmp_path)])
    assert excinfo.value.code == 2
