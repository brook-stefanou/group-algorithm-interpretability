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
