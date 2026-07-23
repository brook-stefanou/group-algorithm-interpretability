"""The opt-in W&B publisher (instruments/publish.py + the `publish` subcommand).

Network-free throughout, in the suite's established style: the W&B boundary is
a fake module injected through the publisher's ``wandb_module`` seam (library
tests) or a monkeypatched ``_import_wandb`` (CLI tests). The fake has no
``Artifact`` attribute at all, so any attempt to touch artifact storage --
which the project is abandoning -- fails loudly rather than passing silently.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from group_algorithm_interp.config import ExperimentConfig, LoggingConfig, ProjectConfig
from group_algorithm_interp.experiment import GroupGeneralizationExperiment
from group_algorithm_interp.instruments import publish as publish_module
from group_algorithm_interp.instruments.publish import (
    OCCUPANCY_JOB_TYPE,
    POOLED_JOB_TYPE,
    publish_records,
    publish_skip_reason,
)
from group_algorithm_interp.instruments.report import measure_run, pool_records

# ---------------------------------------------------------------------------
# campaign-metadata helpers (pair-partner parsing, seed formatting, lookup)
# ---------------------------------------------------------------------------


def test_parse_pair_partner_matches_stream_runs_behaviour():
    assert publish_module._parse_pair_partner(
        "C1 tier-1 pair with (64,80) -- first clean falsifier in the even-order world"
    ) == (64, 80)
    assert publish_module._parse_pair_partner(None) is None
    assert (
        publish_module._parse_pair_partner("smoke cell -- must complete before core starts") is None
    )


def test_format_seed_zero_pads():
    assert publish_module._format_seed(2) == "02"
    assert publish_module._format_seed(49) == "49"


def test_campaign_context_degrades_without_a_match():
    context = publish_module._campaign_context(4, 1, 16, {})
    assert context == {
        "phase": "unknown",
        "cell_name": None,
        "pair_partner_order": None,
        "pair_partner_index": None,
    }


def test_campaign_context_reads_the_matched_cell():
    lookup = {
        (8, 3, 32): {
            "order": 8,
            "index": 3,
            "width": 32,
            "name": "D4",
            "phase": "core",
            "note": "C1 tier-1 pair with (8,4)",
        }
    }
    context = publish_module._campaign_context(8, 3, 32, lookup)
    assert context == {
        "phase": "core",
        "cell_name": "D4",
        "pair_partner_order": 8,
        "pair_partner_index": 4,
    }


def test_load_campaign_lookup_missing_file(tmp_path):
    assert publish_module._load_campaign_lookup(tmp_path / "nope.yaml") == {}


# ---------------------------------------------------------------------------
# _grok_fields -- deriving grokked/censored/epochs_to_grok post hoc for an
# already-finished run. Primary source is the durable selection.json (the
# prune-time grok summary); run.log is a fallback only when it is absent.
# ---------------------------------------------------------------------------


def _write_selection_run_dir(
    parent: Path,
    name: str,
    *,
    selection: dict[str, Any],
    generalize_metric: str = "unleaked_accuracy",
) -> Path:
    run_dir = parent / name
    run_dir.mkdir()
    (run_dir / "selection.json").write_text(json.dumps(selection))
    (run_dir / "manifest.yaml").write_text(
        f"dataset:\n  leakage:\n    generalize_metric: {generalize_metric}\n"
    )
    return run_dir


def _write_categories_run_dir(
    parent: Path,
    name: str,
    *,
    censored: bool,
    leaked_onset_epoch: int | None,
    unleaked_onset_epoch: int | None,
    unleaked_metric_key: str | None = "val/unleaked_accuracy",
) -> Path:
    """A run dir with the ship_runs.py `categories` selection.json shape (the
    majority of the real campaign)."""
    run_dir = parent / name
    run_dir.mkdir()

    def _onset(epoch: int | None, metric_key: str | None) -> dict[str, Any] | None:
        if epoch is None:
            return None
        return {"epoch": epoch, "filename": f"step_{epoch}.pt", "metric_key": metric_key}

    selection = {
        "run_id": name,
        "leaked_metric_key": "val/accuracy",
        "unleaked_metric_key": unleaked_metric_key,
        "threshold": 0.99,
        "censored": censored,
        "interrupted": False,
        "categories": {
            "first": {"epoch": 0, "filename": "step_0.pt"},
            "intermediate": [],
            "last": {"epoch": 59999, "filename": "final.pt"},
            "leaked_onset": _onset(leaked_onset_epoch, "val/accuracy"),
            "unleaked_onset": _onset(unleaked_onset_epoch, unleaked_metric_key),
            "stable_end": _onset(leaked_onset_epoch, "val/accuracy"),
        },
        "anomalies": [],
    }
    (run_dir / "selection.json").write_text(json.dumps(selection))
    return run_dir


def _write_run_log_dir(
    parent: Path, name: str, *, series: list[tuple[int, float]], ceiling: int
) -> Path:
    run_dir = parent / name
    run_dir.mkdir()
    (run_dir / "resolved_config.yaml").write_text(f"optim:\n  epochs: {ceiling}\n")
    lines = [f"epoch {epoch} | {{'val/unleaked_accuracy': {value}}}\n" for epoch, value in series]
    (run_dir / "run.log").write_text("".join(lines))
    return run_dir


def test_grok_fields_reads_the_unleaked_cross_epoch_from_selection_json(tmp_path):
    """Unleaked is the study's canonical generalisation metric, so grok maps
    to the unleaked fields even when the leaked ones would say otherwise."""
    run_dir = _write_selection_run_dir(
        tmp_path,
        "grokked-run",
        selection={
            "censored_leaked": False,
            "censored_unleaked": False,
            "leaked_cross_epoch": 26228,
            "unleaked_cross_epoch": 27142,
        },
    )
    assert publish_module._grok_fields(run_dir) == {
        "grokked": True,
        "censored": False,
        "epochs_to_grok": 27142,
    }


def test_grok_fields_reports_censored_from_selection_json(tmp_path):
    """A run censored on the unleaked metric is censored, whatever the leaked
    metric did -- and carries no epoch (null unleaked_cross_epoch)."""
    run_dir = _write_selection_run_dir(
        tmp_path,
        "censored-run",
        selection={
            "censored_leaked": False,
            "censored_unleaked": True,
            "leaked_cross_epoch": 30000,
            "unleaked_cross_epoch": None,
        },
    )
    assert publish_module._grok_fields(run_dir) == {
        "grokked": False,
        "censored": True,
        "epochs_to_grok": None,
    }


def test_grok_fields_uses_leaked_fields_when_unleaked_subset_was_empty(tmp_path):
    """When the manifest declares the raw-accuracy fallback (empty unleaked
    subset), the leaked grok fields are canonical -- matching the metric
    training itself used."""
    run_dir = _write_selection_run_dir(
        tmp_path,
        "raw-fallback-run",
        selection={
            "censored_leaked": False,
            "censored_unleaked": True,
            "leaked_cross_epoch": 1234,
            "unleaked_cross_epoch": None,
        },
        generalize_metric="raw_test_accuracy",
    )
    assert publish_module._grok_fields(run_dir) == {
        "grokked": True,
        "censored": False,
        "epochs_to_grok": 1234,
    }


def test_grok_fields_reads_unleaked_onset_from_categories_shape(tmp_path):
    """ship_runs.py's `categories` shape: canonical grok comes from the
    unleaked onset snapshot, not the leaked-based top-level `censored`."""
    run_dir = _write_categories_run_dir(
        tmp_path,
        "grokked-run",
        censored=False,
        leaked_onset_epoch=35800,
        unleaked_onset_epoch=37400,
    )
    assert publish_module._grok_fields(run_dir) == {
        "grokked": True,
        "censored": False,
        "epochs_to_grok": 37400,
    }


def test_grok_fields_categories_unleaked_censored_despite_leaked_grok(tmp_path):
    """A run that crossed the leaked bar (top-level censored False) but never
    the unleaked one is censored on the canonical unleaked metric."""
    run_dir = _write_categories_run_dir(
        tmp_path,
        "leaked-only-run",
        censored=False,
        leaked_onset_epoch=35800,
        unleaked_onset_epoch=None,
    )
    assert publish_module._grok_fields(run_dir) == {
        "grokked": False,
        "censored": True,
        "epochs_to_grok": None,
    }


def test_grok_fields_categories_fully_censored(tmp_path):
    run_dir = _write_categories_run_dir(
        tmp_path,
        "censored-run",
        censored=True,
        leaked_onset_epoch=None,
        unleaked_onset_epoch=None,
    )
    assert publish_module._grok_fields(run_dir) == {
        "grokked": False,
        "censored": True,
        "epochs_to_grok": None,
    }


def test_grok_fields_categories_uses_leaked_when_unleaked_subset_empty(tmp_path):
    """Null unleaked_metric_key (empty unleaked subset) makes the leaked onset
    canonical, using the leaked-based top-level censored."""
    run_dir = _write_categories_run_dir(
        tmp_path,
        "raw-fallback-run",
        censored=False,
        leaked_onset_epoch=1234,
        unleaked_onset_epoch=None,
        unleaked_metric_key=None,
    )
    assert publish_module._grok_fields(run_dir) == {
        "grokked": True,
        "censored": False,
        "epochs_to_grok": 1234,
    }


def test_grok_fields_falls_back_to_run_log_without_selection_json(tmp_path):
    """No selection.json (an un-shipped local run) -> the run.log sustained
    onset rule, the live sidecar's approximation."""
    series = [(e, 0.5) for e in range(5)] + [(e, 0.995) for e in range(5, 12)]
    run_dir = _write_run_log_dir(tmp_path, "unshipped-run", series=series, ceiling=100)
    assert publish_module._grok_fields(run_dir) == {
        "grokked": True,
        "censored": False,
        "epochs_to_grok": 5,
    }


def test_grok_fields_is_none_without_selection_json_or_run_log(tmp_path):
    run_dir = tmp_path / "bare"
    run_dir.mkdir()
    (run_dir / "resolved_config.yaml").write_text("optim:\n  epochs: 100\n")
    assert publish_module._grok_fields(run_dir) is None


def test_grok_fields_is_none_for_a_missing_run_dir(tmp_path):
    assert publish_module._grok_fields(tmp_path / "does-not-exist") is None


_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "measure_occupancy.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("measure_occupancy_publish", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["measure_occupancy_publish"] = module
    spec.loader.exec_module(module)
    return module


class _FakeRun:
    """Records everything the publisher does to a run. ``summary`` is a plain
    dict because the publisher only item-assigns into it."""

    def __init__(self, kwargs: dict[str, Any]):
        self.init_kwargs = kwargs
        self.summary: dict[str, Any] = {}
        self.logged: list[dict[str, Any]] = []
        self.defined_metrics: list[tuple[str, dict[str, Any]]] = []
        self.finished = False

    def define_metric(self, name: str, **kwargs: Any) -> None:
        self.defined_metrics.append((name, kwargs))

    def log(self, metrics: dict[str, Any], **_: Any) -> None:
        self.logged.append(metrics)

    def finish(self) -> None:
        self.finished = True


class _FakeWandb:
    """Deliberately exposes ONLY ``init``: no ``Artifact``, no ``Table``, no
    ``log_artifact`` anywhere -- a publisher reaching for artifact storage
    raises AttributeError instead of silently uploading."""

    def __init__(self) -> None:
        self.runs: list[_FakeRun] = []

    def init(self, **kwargs: Any) -> _FakeRun:
        run = _FakeRun(kwargs)
        self.runs.append(run)
        return run


@pytest.fixture(scope="module")
def measured(tmp_path_factory) -> tuple[list[dict[str, Any]], list[Path]]:
    """Two real measured records (seeds 0 and 1 of one config) plus one
    skipped record, produced by the real measure path on short runs; returns
    ``(records, run_dirs)``."""
    root = tmp_path_factory.mktemp("publish-runs")
    run_dirs = []
    for seed in (0, 1):
        config = ProjectConfig(
            device="cpu",
            seed=seed,
            data={"group": "C4", "train_frac": 0.5, "split_seed": 0},
            model={"d_model": 16, "d_mlp": 32, "n_heads": 1},
            optim={"epochs": 8, "log_every": 1, "print_every": 1},
            snapshot={"enabled": False, "final_window_epochs": 2},
            logging=LoggingConfig(mode="disabled"),
            experiment=ExperimentConfig(name="occ-publish"),
        )
        experiment = GroupGeneralizationExperiment(config, runs_root=root)
        experiment.execute()
        run_dirs.append(experiment.run_dir)
    records = [measure_run(run_dir, threshold=0.0) for run_dir in run_dirs]
    records.append(measure_run(run_dirs[0], threshold=0.99))  # skipped
    for run_dir, record in zip(run_dirs, records, strict=False):
        (run_dir / "analysis").mkdir(exist_ok=True)
        (run_dir / "analysis" / "occupancy.json").write_text(json.dumps(record))
    return records, run_dirs


# ---------------------------------------------------------------------------
# The credential gate
# ---------------------------------------------------------------------------


def test_skip_reason_without_api_key(monkeypatch):
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    reason = publish_skip_reason()
    assert reason is not None and "WANDB_API_KEY" in reason


def test_skip_reason_with_disabled_mode(monkeypatch):
    """conftest forces WANDB_MODE=disabled process-wide; even with a key set,
    the publisher must honour it -- publishing is online-only."""
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    reason = publish_skip_reason()
    assert reason is not None and "WANDB_MODE" in reason


def test_no_skip_with_key_and_online_mode(monkeypatch):
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    assert publish_skip_reason() is None


# ---------------------------------------------------------------------------
# publish_records (library, fake W&B injected)
# ---------------------------------------------------------------------------


def test_publish_records_creates_identifiable_metrics_only_runs(measured):
    records, _ = measured
    fake = _FakeWandb()
    counts = publish_records(
        records,
        pool_records(records),
        project="test-project",
        entity="test-entity",
        tags=["occupancy-v1"],
        wandb_module=fake,
    )
    assert counts == {"published": 2, "pooled_published": 1, "skipped_records": 1}
    assert len(fake.runs) == 3

    per_run = fake.runs[:2]
    for run, record in zip(per_run, records[:2], strict=True):
        kwargs = run.init_kwargs
        order, index = record["group"]["order"], record["group"]["index"]
        width = record["model"]["d_model"]
        # Identifiable: attached to the training run's own run_id, and
        # resume="allow" updates a replayed run of the same id in place. No
        # campaign cell matches (4,1) here, so the name degrades the same way
        # the live sidecar's does on an unmatched cell.
        assert kwargs["id"] == record["run_id"]
        assert (
            kwargs["name"] == f"({order},{index}) ({order},{index}) w{width} s{record['seed']:02d}"
        )
        assert kwargs["resume"] == "allow"
        assert kwargs["job_type"] == OCCUPANCY_JOB_TYPE
        assert kwargs["group"] == record["provenance"]["config_group_hash"]
        assert kwargs["project"] == "test-project"
        assert kwargs["entity"] == "test-entity"
        assert kwargs["tags"] == ["occupancy-v1", "campaign-v2", "unknown", record["group"]["name"]]
        assert kwargs["config"]["config_hash"] == record["provenance"]["config_hash"]
        assert kwargs["config"]["phase"] == "unknown"
        assert kwargs["config"]["cell_name"] is None
        assert kwargs["config"]["pair_partner_order"] is None
        assert run.finished

        # Per-block metrics against the block-index step metric, one log call
        # per isotypic block -- plain metrics, no Table, no artifact.
        n_blocks = len(record["analytic_null"])
        assert len(run.logged) == n_blocks
        assert [m["occupancy/block"] for m in run.logged] == list(range(n_blocks))
        assert run.logged[0].keys() >= {"occupancy/left", "occupancy/right", "occupancy/null"}
        assert ("occupancy/block", {}) in run.defined_metrics

        # Headline summary scalars, including the checkpoint substitution.
        assert run.summary["status"] == "measured"
        assert run.summary["checkpoint/substitution"] is None
        assert (
            run.summary["occupancy/left/nontrivial/tv_to_null"]
            == record["occupancy"]["left"]["nontrivial"]["tv_to_null"]
        )

    pooled_run = fake.runs[2]
    assert pooled_run.init_kwargs["job_type"] == POOLED_JOB_TYPE
    assert pooled_run.init_kwargs["id"].startswith("occupancy-pooled-")
    assert pooled_run.init_kwargs["tags"] == [
        "occupancy-v1",
        "campaign-v2",
        "unknown",
        "SmallGroup(4,1)",
    ]
    assert "pooled (2 seeds)" in pooled_run.init_kwargs["name"]
    assert pooled_run.summary["n_runs"] == 2
    assert pooled_run.summary["n_units"] == 64


def test_publish_records_enriches_name_tags_and_config_from_a_campaign_cell(tmp_path, measured):
    """When ``(order, index, width)`` matches a cell, the published run gets
    the same enrichment the live sidecar gives a matched cell: the cell name
    in the display name, a phase tag, and the declared pair partner."""
    records, _ = measured
    record = records[0]
    order, index = record["group"]["order"], record["group"]["index"]
    width = record["model"]["d_model"]
    campaign_config = tmp_path / "core.yaml"
    campaign_config.write_text(
        f"""
cells:
  - phase: core
    order: {order}
    index: {index}
    name: "D4-test"
    width: {width}
    seeds: "0:2"
    note: "C1 tier-1 pair with ({order},99); synthetic test cell"
"""
    )
    fake = _FakeWandb()
    publish_records([record], project="p", wandb_module=fake, campaign_config=campaign_config)
    kwargs = fake.runs[0].init_kwargs
    assert kwargs["name"] == f"D4-test ({order},{index}) w{width} s{record['seed']:02d}"
    assert kwargs["tags"] == ["campaign-v2", "core", "D4-test"]
    assert kwargs["config"]["phase"] == "core"
    assert kwargs["config"]["cell_name"] == "D4-test"
    assert kwargs["config"]["pair_partner_order"] == order
    assert kwargs["config"]["pair_partner_index"] == 99


def test_publish_records_never_touches_artifact_apis(measured):
    """The fake exposes only init/log/summary/finish; a publisher calling
    wandb.Artifact, run.log_artifact, or wandb.Table would raise. Passing at
    all is the assertion, made explicit here."""
    records, _ = measured
    fake = _FakeWandb()
    publish_records(records, None, project="p", wandb_module=fake)
    assert not hasattr(fake, "Artifact")
    assert not hasattr(fake, "Table")
    for run in fake.runs:
        assert not hasattr(run, "log_artifact")


def test_publish_records_adds_grok_fields_read_from_runs_root(tmp_path, measured):
    """The published per-run summary carries the same grokked/censored/
    epochs_to_grok fields the live sidecar writes, derived here from a durable
    selection.json found under an explicit runs_root -- exercising the whole
    wiring from publish_records down to _grok_fields."""
    records, _ = measured
    record = records[0]
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    _write_selection_run_dir(
        runs_root,
        str(record["run_id"]),
        selection={
            "censored_leaked": False,
            "censored_unleaked": False,
            "leaked_cross_epoch": 26228,
            "unleaked_cross_epoch": 27142,
        },
    )

    fake = _FakeWandb()
    publish_records([record], project="p", wandb_module=fake, runs_root=runs_root)
    summary = fake.runs[0].summary
    assert summary["grokked"] is True
    assert summary["censored"] is False
    assert summary["epochs_to_grok"] == 27142


def test_publish_records_omits_grok_fields_without_a_matching_run_dir(tmp_path, measured):
    """No matching run directory under runs_root (an empty root here) degrades
    to simply omitting the fields -- never a failure."""
    records, _ = measured
    empty_runs_root = tmp_path / "empty-runs"
    empty_runs_root.mkdir()
    fake = _FakeWandb()
    publish_records([records[0]], project="p", wandb_module=fake, runs_root=empty_runs_root)
    assert "grokked" not in fake.runs[0].summary
    assert "censored" not in fake.runs[0].summary
    assert "epochs_to_grok" not in fake.runs[0].summary


def test_publish_records_skips_unmeasured_records(measured):
    records, _ = measured
    fake = _FakeWandb()
    skipped_only = [r for r in records if r["status"] == "skipped"]
    counts = publish_records(skipped_only, None, project="p", wandb_module=fake)
    assert counts == {"published": 0, "pooled_published": 0, "skipped_records": 1}
    assert fake.runs == []


# ---------------------------------------------------------------------------
# The CLI subcommand
# ---------------------------------------------------------------------------


def test_cli_publish_does_nothing_without_api_key(monkeypatch, capsys, tmp_path):
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    script = _load_script()
    code = script.main(["publish", str(tmp_path)])  # dir contents never read
    assert code == 0
    out = capsys.readouterr().out
    assert "not publishing" in out and "WANDB_API_KEY" in out


def test_cli_publish_pushes_measured_run_dirs(measured, monkeypatch, capsys):
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    fake = _FakeWandb()
    import group_algorithm_interp.instruments.publish as publish_module

    monkeypatch.setattr(publish_module, "_import_wandb", lambda: fake)
    script = _load_script()

    _, run_dirs = measured
    code = script.main(["publish", *(str(d) for d in run_dirs), "--tags", "smoke, v1"])
    assert code == 0
    assert len(fake.runs) == 3  # 2 per-run + 1 pooled
    assert fake.runs[0].init_kwargs["tags"][:2] == ["smoke", "v1"]
    assert "campaign-v2" in fake.runs[0].init_kwargs["tags"]
    assert "published 2 run(s), 1 pooled" in capsys.readouterr().out


def test_cli_publish_requires_measurement_first(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    script = _load_script()
    empty_run = tmp_path / "not-measured"
    empty_run.mkdir()
    code = script.main(["publish", str(empty_run)])
    assert code == 1
    assert "run `measure` first" in capsys.readouterr().out
