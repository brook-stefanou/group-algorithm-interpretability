"""Focused tests for `scripts/sync_runs.py`.

No network, no W&B keys, no real `rclone` binary: the W&B boundary is mocked
the same way `tests/test_wandb_utils.py` mocks it (monkeypatching `wandb.*`),
and the rclone boundary is exercised through the injectable `run` callable
`sync_rclone` accepts (mirroring `scripts/preflight.py`'s `check_sage`
pattern). `conftest.py` forces `WANDB_MODE=disabled` process-wide, which is
exactly the case this module must handle without ever importing `wandb`.

The script is an executable entry point, not part of the installed package,
so it is loaded by file path (mirrors `tests/test_run_batch_command.py`
and `tests/test_preflight.py`).
"""

from __future__ import annotations

import argparse
import builtins
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from group_algorithm_interp.config import LoggingConfig, ProjectConfig
from group_algorithm_interp.manifest import (
    create_manifest,
    create_run_dir,
    finalize_manifest,
    read_manifest,
    save_resolved_config,
)

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "sync_runs.py"


def _load_sync_runs() -> ModuleType:
    spec = importlib.util.spec_from_file_location("sync_runs", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sync_runs = _load_sync_runs()


def _make_run(
    runs_dir: Path,
    run_id: str,
    *,
    status: str = "completed",
    with_config: bool = True,
    with_log: bool = True,
    with_final_checkpoint: bool = True,
    with_snapshots: bool = True,
) -> Path:
    """A realistic run directory: real manifest/config machinery, plus the
    files a real training run would have written alongside them."""
    run_dir = create_run_dir(run_id, root=runs_dir)
    config = ProjectConfig(logging=LoggingConfig(mode="disabled"))
    create_manifest(config, run_dir)
    if status != "running":
        finalize_manifest(run_dir, status=status, summary={"completed_steps": 10, "metrics": {}})
    if with_config:
        save_resolved_config(config, run_dir)
    if with_log:
        (run_dir / "run.log").write_text("epoch 0 | train/loss=1.0\n")
    if with_final_checkpoint:
        (run_dir / "checkpoints" / "final.pt").write_bytes(b"final-checkpoint")
    if with_snapshots:
        (run_dir / "checkpoints" / "step_0.pt").write_bytes(b"snap-0")
        (run_dir / "checkpoints" / "generalized_step_5.pt").write_bytes(b"snap-5")
    return run_dir


# --- discovery + completeness ------------------------------------------------


def test_discover_runs_ignores_non_run_directories(tmp_path):
    runs_dir = tmp_path / "runs"
    _make_run(runs_dir, "real-run")
    (runs_dir / "schedules").mkdir(parents=True)
    (runs_dir / "schedules" / "schedule_x.yaml").write_text("kind: gpu_seed_schedule\n")
    (runs_dir / ".sync_ledger.json").write_text("{}")

    assert [p.name for p in sync_runs.discover_runs(runs_dir)] == ["real-run"]


def test_discover_runs_on_missing_directory_is_empty(tmp_path):
    assert sync_runs.discover_runs(tmp_path / "does-not-exist") == []


def test_collect_run_records_skips_running_and_keeps_terminal_statuses(tmp_path):
    runs_dir = tmp_path / "runs"
    _make_run(runs_dir, "run-running", status="running")
    _make_run(runs_dir, "run-completed", status="completed")
    _make_run(runs_dir, "run-failed", status="failed")
    _make_run(runs_dir, "run-aborted", status="aborted")

    records = sync_runs.collect_run_records(runs_dir, sync_runs.ALL_CATEGORIES)
    ids = {r.run_id for r in records}
    assert ids == {"run-completed", "run-failed", "run-aborted"}
    statuses = {r.run_id: r.status for r in records}
    assert statuses == {
        "run-completed": "completed",
        "run-failed": "failed",
        "run-aborted": "aborted",
    }


def test_collect_run_records_skips_unreadable_manifest_and_continues(tmp_path):
    runs_dir = tmp_path / "runs"
    _make_run(runs_dir, "good")
    bad_dir = runs_dir / "bad"
    bad_dir.mkdir(parents=True)
    # Valid YAML, but not a mapping -- read_manifest raises ValueError on this.
    (bad_dir / "manifest.yaml").write_text("- not\n- a\n- mapping\n")

    warnings: list[str] = []
    records = sync_runs.collect_run_records(
        runs_dir, sync_runs.ALL_CATEGORIES, warn=warnings.append
    )

    assert [r.run_id for r in records] == ["good"]
    assert warnings and "bad" in warnings[0]


# --- file-set selection -------------------------------------------------------


def test_resolve_included_files_all_categories(tmp_path):
    runs_dir = tmp_path / "runs"
    run_dir = _make_run(runs_dir, "r1")

    files = sync_runs.resolve_included_files(run_dir, sync_runs.ALL_CATEGORIES)

    assert set(files) == set(sync_runs.ALL_CATEGORIES)
    assert files["manifest"] == [run_dir / "manifest.yaml"]
    assert files["config"] == [run_dir / "resolved_config.yaml"]
    assert files["summary"] == [run_dir / "run.log"]
    assert files["checkpoint"] == [run_dir / "checkpoints" / "final.pt"]
    snapshot_names = {p.name for p in files["snapshots"]}
    assert snapshot_names == {"step_0.pt", "generalized_step_5.pt"}
    assert "final.pt" not in snapshot_names  # final.pt is its own category, never double-counted


def test_resolve_included_files_light_categories(tmp_path):
    runs_dir = tmp_path / "runs"
    run_dir = _make_run(runs_dir, "r1")

    files = sync_runs.resolve_included_files(run_dir, sync_runs.LIGHT_CATEGORIES)

    assert set(files) == {"manifest", "summary"}


def test_resolve_included_files_missing_files_resolve_empty_not_error(tmp_path):
    runs_dir = tmp_path / "runs"
    run_dir = _make_run(
        runs_dir, "r1", with_config=False, with_final_checkpoint=False, with_snapshots=False
    )

    files = sync_runs.resolve_included_files(run_dir, sync_runs.ALL_CATEGORIES)

    assert files["config"] == []
    assert files["checkpoint"] == []
    assert files["snapshots"] == []
    assert files["manifest"] != []  # manifest always exists once a run is created


def test_parse_categories_rejects_unknown_category():
    with pytest.raises(argparse.ArgumentTypeError):
        sync_runs._parse_categories("manifest,bogus")


def test_parse_categories_dedupes_and_sorts():
    assert sync_runs._parse_categories("summary, manifest ,summary") == ("manifest", "summary")


def test_light_and_include_flags_are_mutually_exclusive():
    parser = sync_runs.build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--light", "--include", "manifest"])


# --- ledger idempotency --------------------------------------------------------


def test_already_synced_is_a_category_superset_check():
    ledger = {"runs": {}}
    key = sync_runs.transport_key("rclone", "remote:bucket")

    assert not sync_runs.already_synced(ledger, "r1", key, ["manifest", "summary"])

    sync_runs.record_synced(ledger, "r1", key, ["manifest", "summary"], "2026-07-15T00:00:00Z")
    assert sync_runs.already_synced(ledger, "r1", key, ["manifest"])
    assert sync_runs.already_synced(ledger, "r1", key, ["manifest", "summary"])
    assert not sync_runs.already_synced(ledger, "r1", key, ["manifest", "summary", "snapshots"])

    # Requesting a wider set later (e.g. dropping --light) is a real sync, and
    # the ledger must accumulate rather than overwrite what was already sent.
    sync_runs.record_synced(ledger, "r1", key, ["snapshots"], "2026-07-15T00:05:00Z")
    assert sync_runs.already_synced(ledger, "r1", key, ["manifest", "summary", "snapshots"])


def test_transport_key_namespaces_by_destination():
    """A run already synced to one rclone target must not be considered synced
    for a different one -- switching targets is a real upload, not a repeat."""
    assert sync_runs.transport_key("rclone", "a:bucket") != sync_runs.transport_key(
        "rclone", "b:bucket"
    )


def test_ledger_round_trips_through_disk(tmp_path):
    path = tmp_path / "runs" / ".sync_ledger.json"
    path.parent.mkdir(parents=True)

    ledger = sync_runs.load_ledger(path)
    assert ledger == {"runs": {}}

    sync_runs.record_synced(ledger, "r1", "wandb:/proj", ["manifest"], "2026-07-15T00:00:00Z")
    sync_runs.save_ledger(path, ledger)

    reloaded = sync_runs.load_ledger(path)
    assert reloaded["runs"]["r1"]["wandb:/proj"]["categories"] == ["manifest"]


def test_load_ledger_rejects_a_malformed_file(tmp_path):
    path = tmp_path / "ledger.json"
    path.write_text('{"not_runs": {}}')
    with pytest.raises(ValueError):
        sync_runs.load_ledger(path)


# --- W&B transport -------------------------------------------------------------


def test_wandb_skip_reason_covers_disabled_and_offline(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    assert sync_runs.wandb_skip_reason() == "disabled"
    monkeypatch.setenv("WANDB_MODE", "OFFLINE")
    assert sync_runs.wandb_skip_reason() == "offline"
    monkeypatch.setenv("WANDB_MODE", "online")
    assert sync_runs.wandb_skip_reason() is None
    monkeypatch.delenv("WANDB_MODE", raising=False)
    assert sync_runs.wandb_skip_reason() is None  # unset defaults to "online"


def test_sync_wandb_skips_cleanly_and_does_not_import_wandb_when_disabled(tmp_path, monkeypatch):
    """conftest.py forces WANDB_MODE=disabled for the whole suite -- this is
    the exact case `just gate` (network-free, no API keys) depends on."""
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "wandb":
            raise AssertionError("disabled mode imported wandb")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    runs_dir = tmp_path / "runs"
    run_dir = _make_run(runs_dir, "r1")
    record = sync_runs.RunRecord(
        "r1",
        run_dir,
        "completed",
        read_manifest(run_dir),
        {"manifest": [run_dir / "manifest.yaml"]},
    )

    outcome = sync_runs.sync_wandb([record], project="p", entity=None, dry_run=False)
    assert outcome.skipped is True
    assert outcome.synced == []


def test_sync_wandb_offline_also_skips_cleanly(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    runs_dir = tmp_path / "runs"
    run_dir = _make_run(runs_dir, "r1")
    record = sync_runs.RunRecord("r1", run_dir, "completed", read_manifest(run_dir), {})

    outcome = sync_runs.sync_wandb([record], project="p", entity=None, dry_run=False)
    assert outcome.skipped is True


def test_sync_wandb_dry_run_does_not_import_wandb(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("WANDB_MODE", "online")
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "wandb":
            raise AssertionError("dry-run imported wandb")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    runs_dir = tmp_path / "runs"
    run_dir = _make_run(runs_dir, "r1")
    files = sync_runs.resolve_included_files(run_dir, sync_runs.ALL_CATEGORIES)
    record = sync_runs.RunRecord("r1", run_dir, "completed", read_manifest(run_dir), files)

    outcome = sync_runs.sync_wandb([record], project="p", entity=None, dry_run=True)
    assert outcome.synced == ["r1"]
    assert "[dry-run]" in capsys.readouterr().out


class _FakeArtifact:
    """Mirrors the wandb.Artifact surface sync_wandb touches, including the
    post-commit contract: wait() must be called before qualified_name means
    anything, and a wait() failure is how a broken upload actually surfaces."""

    def __init__(self, name, type, metadata=None):  # noqa: A002 (mirrors wandb's own kwarg)
        self.name = name
        self.type = type
        self.metadata = metadata
        self.added_files: list[tuple[str, str | None]] = []
        self.wait_called = False

    def add_file(self, path, name=None):
        self.added_files.append((path, name))

    def wait(self):
        self.wait_called = True
        return self

    @property
    def qualified_name(self):
        return f"entity/project/{self.name}:v0"


class _FakeEntry:
    def __init__(self, size):
        self.size = size
        self.downloaded = False

    def download(self, root=None):
        self.downloaded = True
        return root


class _FakeFetchedArtifact:
    """What Api().artifact(...) returns during verify-back."""

    def __init__(self, entries):
        class _Manifest:
            pass

        self.manifest = _Manifest()
        self.manifest.entries = entries
        self.downloaded: list[str] = []

    def get_entry(self, name):
        self.downloaded.append(name)
        return self.manifest.entries[name]


def _install_fake_wandb(monkeypatch, *, fetched=None, log_artifact=None):
    """Wire wandb.init/Artifact/Api fakes; returns (logged_artifacts, api_requests)."""
    import wandb

    monkeypatch.setenv("WANDB_MODE", "online")
    monkeypatch.setattr(wandb, "run", None, raising=False)

    logged: list[_FakeArtifact] = []
    requested: list[str] = []

    class _FakeRun:
        def log_artifact(self, artifact):
            if log_artifact is not None:
                log_artifact(artifact)
            logged.append(artifact)

        def finish(self):
            pass

    class _FakeApi:
        def artifact(self, qualified_name, type=None):  # noqa: A002
            requested.append(qualified_name)
            return fetched if fetched is not None else _FakeFetchedArtifact({})

    monkeypatch.setattr(wandb, "init", lambda **kwargs: _FakeRun(), raising=False)
    monkeypatch.setattr(wandb, "Artifact", _FakeArtifact, raising=False)
    monkeypatch.setattr(wandb, "Api", _FakeApi, raising=False)
    return logged, requested


def test_sync_wandb_uploads_artifact_with_manifest_metadata(tmp_path, monkeypatch):
    logged, requested = _install_fake_wandb(monkeypatch)

    runs_dir = tmp_path / "runs"
    run_dir = _make_run(runs_dir, "r1")
    manifest = read_manifest(run_dir)
    files = sync_runs.resolve_included_files(run_dir, sync_runs.ALL_CATEGORIES)
    record = sync_runs.RunRecord("r1", run_dir, "completed", manifest, files)

    outcome = sync_runs.sync_wandb([record], project="p", entity=None, dry_run=False)

    assert outcome.synced == ["r1"]
    assert len(logged) == 1
    assert logged[0].name == "run-r1"
    assert logged[0].metadata == manifest
    names = {name for _, name in logged[0].added_files}
    assert "manifest.yaml" in names
    assert "checkpoints/final.pt" in names
    # the commit was awaited and the committed artifact was fetched back
    assert logged[0].wait_called
    assert requested == ["entity/project/run-r1:v0"]


def test_sync_wandb_one_run_failing_does_not_abort_the_batch(tmp_path, monkeypatch):
    def explode_on_bad(artifact):
        if artifact.name == "run-bad":
            raise RuntimeError("boom")

    _install_fake_wandb(monkeypatch, log_artifact=explode_on_bad)

    runs_dir = tmp_path / "runs"
    good_dir = _make_run(runs_dir, "good")
    bad_dir = _make_run(runs_dir, "bad")
    records = [
        sync_runs.RunRecord("bad", bad_dir, "completed", read_manifest(bad_dir), {}),
        sync_runs.RunRecord("good", good_dir, "completed", read_manifest(good_dir), {}),
    ]

    outcome = sync_runs.sync_wandb(records, project="p", entity=None, dry_run=False)
    assert outcome.synced == ["good"]


def test_sync_wandb_commit_failure_is_not_recorded_as_synced(tmp_path, monkeypatch):
    """A run whose Artifact.wait() raises (async upload failed after
    log_artifact returned) must not land in `synced` -- this is the exact
    2026-07-15 failure shape: every byte 403'd yet the old code recorded
    ledger success and the pruner deleted the local snapshots."""
    import wandb

    logged, _ = _install_fake_wandb(monkeypatch)

    class _FailingWaitArtifact(_FakeArtifact):
        def wait(self):
            raise ValueError("ArtifactSaver.uploadFiles: most remaining uploads have failed")

    monkeypatch.setattr(wandb, "Artifact", _FailingWaitArtifact, raising=False)

    runs_dir = tmp_path / "runs"
    run_dir = _make_run(runs_dir, "r1")
    record = sync_runs.RunRecord("r1", run_dir, "completed", read_manifest(run_dir), {})

    outcome = sync_runs.sync_wandb([record], project="p", entity=None, dry_run=False)
    assert outcome.synced == []


def test_sync_wandb_verify_back_failure_is_not_recorded_as_synced(tmp_path, monkeypatch):
    """Committed but not byte-fetchable (e.g. storage 403s the download) is
    still a failure: the ledger must not absorb it."""

    class _Deny403:
        def __init__(self):
            class _Manifest:
                entries = {"manifest.yaml": _FakeEntry(10)}

            self.manifest = _Manifest()

        def get_entry(self, name):
            raise RuntimeError("403 SignatureDoesNotMatch")

    _install_fake_wandb(monkeypatch, fetched=_Deny403())

    runs_dir = tmp_path / "runs"
    run_dir = _make_run(runs_dir, "r1")
    record = sync_runs.RunRecord("r1", run_dir, "completed", read_manifest(run_dir), {})

    outcome = sync_runs.sync_wandb([record], project="p", entity=None, dry_run=False)
    assert outcome.synced == []


def test_verify_artifact_bytes_downloads_smallest_entry():
    fetched = _FakeFetchedArtifact(
        {
            "checkpoints/final.pt": _FakeEntry(5_000_000),
            "manifest.yaml": _FakeEntry(300),
            "run.log": _FakeEntry(90_000),
        }
    )

    class _Api:
        def artifact(self, qualified_name, type=None):  # noqa: A002
            return fetched

    sync_runs.verify_artifact_bytes(_Api(), "e/p/run-r1:v0")
    assert fetched.downloaded == ["manifest.yaml"]
    assert fetched.manifest.entries["manifest.yaml"].downloaded


def test_verify_artifact_bytes_tolerates_empty_manifest():
    class _Api:
        def artifact(self, qualified_name, type=None):  # noqa: A002
            return _FakeFetchedArtifact({})

    sync_runs.verify_artifact_bytes(_Api(), "e/p/run-r1:v0")  # must not raise


# --- rclone transport ----------------------------------------------------------


def test_sync_rclone_copies_only_requested_categories(tmp_path):
    runs_dir = tmp_path / "runs"
    run_dir = _make_run(runs_dir, "r1")
    files = sync_runs.resolve_included_files(run_dir, ("manifest", "summary"))
    record = sync_runs.RunRecord("r1", run_dir, "completed", read_manifest(run_dir), files)

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        list_path = Path(cmd[cmd.index("--files-from") + 1])
        listed = set(list_path.read_text().splitlines())
        assert listed == {"manifest.yaml", "run.log"}
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    outcome = sync_runs.sync_rclone(
        [record], target="remote:bucket/prefix", dry_run=False, run=fake_run
    )

    assert outcome.synced == ["r1"]
    assert len(calls) == 1
    assert calls[0][:3] == ["rclone", "copy", str(run_dir)]
    assert calls[0][3] == "remote:bucket/prefix/r1"


def test_sync_rclone_strips_trailing_slash_from_target(tmp_path):
    runs_dir = tmp_path / "runs"
    run_dir = _make_run(runs_dir, "r1")
    files = sync_runs.resolve_included_files(run_dir, ("manifest",))
    record = sync_runs.RunRecord("r1", run_dir, "completed", read_manifest(run_dir), files)

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    sync_runs.sync_rclone([record], target="remote:bucket/", dry_run=False, run=fake_run)
    assert calls[0][3] == "remote:bucket/r1"


def test_sync_rclone_failure_is_logged_and_skipped_not_raised(tmp_path, capsys):
    runs_dir = tmp_path / "runs"
    run_dir = _make_run(runs_dir, "r1")
    files = sync_runs.resolve_included_files(run_dir, ("manifest",))
    record = sync_runs.RunRecord("r1", run_dir, "completed", read_manifest(run_dir), files)

    def failing_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom")

    outcome = sync_runs.sync_rclone(
        [record], target="remote:bucket", dry_run=False, run=failing_run
    )
    assert outcome.synced == []
    assert "FAILED" in capsys.readouterr().err


def test_sync_rclone_dry_run_does_not_invoke_subprocess(tmp_path):
    runs_dir = tmp_path / "runs"
    run_dir = _make_run(runs_dir, "r1")
    files = sync_runs.resolve_included_files(run_dir, sync_runs.ALL_CATEGORIES)
    record = sync_runs.RunRecord("r1", run_dir, "completed", read_manifest(run_dir), files)

    def boom(cmd, **kwargs):
        raise AssertionError("subprocess must not run under --dry-run")

    outcome = sync_runs.sync_rclone([record], target="remote:bucket", dry_run=True, run=boom)
    assert outcome.synced == ["r1"]


def test_sync_rclone_skips_run_with_no_resolvable_files(tmp_path):
    runs_dir = tmp_path / "runs"
    run_dir = _make_run(
        runs_dir,
        "r1",
        with_config=False,
        with_log=False,
        with_final_checkpoint=False,
        with_snapshots=False,
    )
    files = sync_runs.resolve_included_files(run_dir, ("config", "checkpoint", "snapshots"))
    record = sync_runs.RunRecord("r1", run_dir, "completed", read_manifest(run_dir), files)

    calls: list[object] = []
    outcome = sync_runs.sync_rclone(
        [record], target="remote:bucket", dry_run=False, run=lambda cmd, **kw: calls.append(cmd)
    )
    assert outcome.synced == []
    assert calls == []


# --- main() orchestration -------------------------------------------------------


def test_main_no_completed_runs_prints_and_exits_zero(tmp_path, capsys):
    runs_dir = tmp_path / "runs"
    _make_run(runs_dir, "r1", status="running")

    exit_code = sync_runs.main(
        ["--runs-dir", str(runs_dir), "--no-wandb", "--rclone-target", "remote:bucket"]
    )
    assert exit_code == 0
    assert "no completed runs" in capsys.readouterr().out


def test_main_errors_when_no_transport_requested(tmp_path):
    runs_dir = tmp_path / "runs"
    with pytest.raises(SystemExit):
        sync_runs.main(["--runs-dir", str(runs_dir), "--no-wandb"])


def test_main_rclone_dry_run_writes_no_ledger(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    _make_run(runs_dir, "r1")
    _make_run(runs_dir, "r2", status="running")

    calls: list[tuple[str, ...]] = []

    def fake_sync_rclone(records, *, target, dry_run, run=None):
        calls.append(tuple(r.run_id for r in records))
        return sync_runs.TransportOutcome(synced=[r.run_id for r in records])

    monkeypatch.setattr(sync_runs, "sync_rclone", fake_sync_rclone)

    exit_code = sync_runs.main(
        [
            "--runs-dir",
            str(runs_dir),
            "--no-wandb",
            "--rclone-target",
            "remote:bucket",
            "--dry-run",
        ]
    )
    assert exit_code == 0
    assert calls == [("r1",)]  # the running run is never a candidate
    assert not (runs_dir / ".sync_ledger.json").exists()


def test_main_rclone_ledger_prevents_reupload(tmp_path, monkeypatch, capsys):
    runs_dir = tmp_path / "runs"
    _make_run(runs_dir, "r1")

    calls: list[tuple[str, ...]] = []

    def fake_sync_rclone(records, *, target, dry_run, run=None):
        calls.append(tuple(r.run_id for r in records))
        return sync_runs.TransportOutcome(synced=[r.run_id for r in records])

    monkeypatch.setattr(sync_runs, "sync_rclone", fake_sync_rclone)
    args = ["--runs-dir", str(runs_dir), "--no-wandb", "--rclone-target", "remote:bucket"]

    assert sync_runs.main(args) == 0
    assert calls == [("r1",)]
    assert (runs_dir / ".sync_ledger.json").is_file()

    assert sync_runs.main(args) == 0
    assert calls == [("r1",)]  # unchanged: the second run never called the transport again
    assert "nothing to sync" in capsys.readouterr().out


def test_main_force_bypasses_the_ledger(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    _make_run(runs_dir, "r1")

    calls: list[tuple[str, ...]] = []

    def fake_sync_rclone(records, *, target, dry_run, run=None):
        calls.append(tuple(r.run_id for r in records))
        return sync_runs.TransportOutcome(synced=[r.run_id for r in records])

    monkeypatch.setattr(sync_runs, "sync_rclone", fake_sync_rclone)
    args = ["--runs-dir", str(runs_dir), "--no-wandb", "--rclone-target", "remote:bucket"]

    sync_runs.main(args)
    sync_runs.main([*args, "--force"])
    assert calls == [("r1",), ("r1",)]


def test_main_light_only_ships_manifest_and_summary(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    _make_run(runs_dir, "r1")

    captured: list[object] = []

    def fake_sync_rclone(records, *, target, dry_run, run=None):
        captured.extend(records)
        return sync_runs.TransportOutcome(synced=[r.run_id for r in records])

    monkeypatch.setattr(sync_runs, "sync_rclone", fake_sync_rclone)

    sync_runs.main(
        [
            "--runs-dir",
            str(runs_dir),
            "--no-wandb",
            "--rclone-target",
            "remote:bucket",
            "--light",
        ]
    )
    assert set(captured[0].files) == {"manifest", "summary"}  # type: ignore[attr-defined]


def test_main_returns_1_when_a_run_fails_to_sync(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    _make_run(runs_dir, "r1")
    _make_run(runs_dir, "r2")

    def fake_sync_rclone(records, *, target, dry_run, run=None):
        return sync_runs.TransportOutcome(synced=[r.run_id for r in records if r.run_id == "r2"])

    monkeypatch.setattr(sync_runs, "sync_rclone", fake_sync_rclone)

    exit_code = sync_runs.main(
        ["--runs-dir", str(runs_dir), "--no-wandb", "--rclone-target", "remote:bucket"]
    )
    assert exit_code == 1


def test_main_rclone_target_from_env_var(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    _make_run(runs_dir, "r1")
    monkeypatch.setenv("GAI_RCLONE_TARGET", "remote:from-env")

    seen_targets: list[str] = []

    def fake_sync_rclone(records, *, target, dry_run, run=None):
        seen_targets.append(target)
        return sync_runs.TransportOutcome(synced=[r.run_id for r in records])

    monkeypatch.setattr(sync_runs, "sync_rclone", fake_sync_rclone)
    sync_runs.main(["--runs-dir", str(runs_dir), "--no-wandb"])
    assert seen_targets == ["remote:from-env"]


def test_main_wandb_skips_cleanly_when_disabled_and_still_runs_rclone(
    tmp_path, monkeypatch, capsys
):
    """WANDB_MODE=disabled (forced by conftest.py) must not crash main(), and
    a simultaneously-requested rclone transport must still run."""
    runs_dir = tmp_path / "runs"
    _make_run(runs_dir, "r1")

    rclone_calls: list[tuple[str, ...]] = []

    def fake_sync_rclone(records, *, target, dry_run, run=None):
        rclone_calls.append(tuple(r.run_id for r in records))
        return sync_runs.TransportOutcome(synced=[r.run_id for r in records])

    monkeypatch.setattr(sync_runs, "sync_rclone", fake_sync_rclone)

    exit_code = sync_runs.main(["--runs-dir", str(runs_dir), "--rclone-target", "remote:bucket"])
    assert exit_code == 0
    assert rclone_calls == [("r1",)]
    assert "WANDB_MODE=disabled" in capsys.readouterr().out
    # A skipped wandb sync must not write a ledger entry for the wandb transport.
    ledger = sync_runs.load_ledger(runs_dir / ".sync_ledger.json")
    assert "wandb:/group-algorithm-interp" not in ledger["runs"]["r1"]
