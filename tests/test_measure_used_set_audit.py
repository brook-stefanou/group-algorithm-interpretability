"""The used-set-audit entry point (scripts/measure_used_set_audit.py).

Runs the real ``measure`` command against a short training run on the
fixture corpus -- offline, no W&B, no network. The script is an executable
entry point, not part of the installed package, so it is loaded by file path
(mirrors ``tests/test_measure_gcr_readout.py``). The I-15 dependency this
driver reads is a real record, produced by
``scripts/measure_isotypic_ablation.py``'s own functions against the same
training run -- not a hand-built stub -- so the ``causally_used_blocks``
derivation is exercised end to end.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from group_algorithm_interp.config import ExperimentConfig, LoggingConfig, ProjectConfig
from group_algorithm_interp.experiment import GroupGeneralizationExperiment
from group_algorithm_interp.instruments.coset import ABLATION_MODES

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "measure_used_set_audit.py"
_ISO_SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "measure_isotypic_ablation.py"
)

EPOCHS = 10


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_script() -> ModuleType:
    return _load_module("measure_used_set_audit", _SCRIPT_PATH)


def _load_iso_script() -> ModuleType:
    return _load_module("measure_isotypic_ablation_for_used_set_audit", _ISO_SCRIPT_PATH)


def _train(runs_root: Path, *, width: int = 16, seed: int = 0) -> Path:
    config = ProjectConfig(
        device="cpu",
        seed=seed,
        data={"group": {"order": 8, "index": 3}, "train_frac": 0.5, "split_seed": 0},
        model={"d_model": width, "d_mlp": 2 * width, "n_heads": 1},
        optim={"epochs": EPOCHS, "log_every": 1, "print_every": 1},
        snapshot={
            "enabled": True,
            "interval": 4,
            "log_dense_until": 4,
            "event_based": False,
            "final_window_epochs": 3,
        },
        logging=LoggingConfig(mode="disabled"),
        experiment=ExperimentConfig(name="used-set-audit-measure"),
    )
    experiment = GroupGeneralizationExperiment(config, runs_root=runs_root)
    experiment.execute()
    return experiment.run_dir


@pytest.fixture(scope="module")
def dihedral_run(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("used-set-audit-runs")
    return _train(root)


@pytest.fixture(scope="module")
def iso_ablation_dir(dihedral_run, tmp_path_factory) -> Path:
    """A real I-15 per-cell record (``scripts/measure_isotypic_ablation.py``'s
    own ``_measure_run``/``aggregate_cell``), written to
    ``<dir>/iso_ablation_8_3_w16.json`` -- exactly the file
    ``cell_used_blocks`` reads."""
    iso_module = _load_iso_script()
    record = iso_module._measure_run(dihedral_run, metric="val/accuracy", threshold=0.0, n_random=2)
    aggregate = iso_module.aggregate_cell([record], order=8, index=3, width=16, name="D8")
    coset_dir = tmp_path_factory.mktemp("used-set-audit-coset")
    (coset_dir / "iso_ablation_8_3_w16.json").write_text(json.dumps(aggregate))
    return coset_dir


def test_cell_used_blocks_is_pending_when_the_i15_file_is_absent(tmp_path):
    script = _load_script()
    dependency = script.cell_used_blocks(8, 3, 16, coset_dir=tmp_path / "nowhere")
    assert dependency["status"] == "pending"
    assert len(dependency["missing"]) == 1
    assert "iso_ablation_8_3_w16.json" in dependency["missing"][0]


def test_cell_used_blocks_derives_the_used_set_from_a_real_i15_record(iso_ablation_dir):
    script = _load_script()
    dependency = script.cell_used_blocks(8, 3, 16, coset_dir=iso_ablation_dir)
    assert dependency["status"] == "measured"
    assert dependency["iso_ablation_path"].endswith("iso_ablation_8_3_w16.json")
    assert isinstance(dependency["used_blocks"], list)
    # Every returned index must be an occupied (non-trivial) isotypic block.
    iso_record = json.loads((iso_ablation_dir / "iso_ablation_8_3_w16.json").read_text())
    valid_indices = {b["block_index"] for b in iso_record["blocks"]}
    assert set(dependency["used_blocks"]) <= valid_indices


def test_used_set_audit_run_produces_a_full_record(dihedral_run, iso_ablation_dir):
    script = _load_script()
    dependency = script.cell_used_blocks(8, 3, 16, coset_dir=iso_ablation_dir)
    used_blocks = dependency["used_blocks"] or [
        b["block_index"]
        for b in json.loads((iso_ablation_dir / "iso_ablation_8_3_w16.json").read_text())["blocks"]
    ]
    record = script.used_set_audit_run(
        dihedral_run,
        used_blocks,
        threshold=0.0,
        n_random=2,
        near_full_rank_fraction=0.9,
        min_flip_over_random=0.05,
        sufficiency_retention_fraction=0.9,
        minimality_mode="zero",
    )
    assert record["status"] == "measured"
    assert record["used_blocks"] == list(used_blocks)
    usa = record["used_set_audit"]
    assert usa["instrument"] == "used-set-causal-sufficiency-audit"
    assert "verdict" in usa
    assert set(usa["completeness"]["accuracy_retention_fraction"]) == set(ABLATION_MODES)

    provenance = record["provenance"]
    assert provenance["config_hash"]
    assert provenance["checkpoint_sha256"]
    assert provenance["instrument_code_sha256"]
    assert provenance["group_artifact_path"].endswith("smallgroup_8_3.npz")
    json.dumps(record)  # the whole record must be JSON-serialisable


def test_used_set_audit_run_skips_a_run_with_no_stable_checkpoint(dihedral_run):
    script = _load_script()
    record = script.used_set_audit_run(
        dihedral_run,
        [1],
        threshold=0.99,
        n_random=2,
        near_full_rank_fraction=0.9,
        min_flip_over_random=0.05,
        sufficiency_retention_fraction=0.9,
        minimality_mode="zero",
    )
    assert record["status"] == "skipped"
    assert record["checkpoint_selection"]["reason"] is not None
    assert "used_set_audit" not in record


def test_summarise_cell_aggregates_the_verdicts(dihedral_run, iso_ablation_dir):
    script = _load_script()
    dependency = script.cell_used_blocks(8, 3, 16, coset_dir=iso_ablation_dir)
    used_blocks = dependency["used_blocks"] or [1]
    record = script.used_set_audit_run(
        dihedral_run,
        used_blocks,
        threshold=0.0,
        n_random=2,
        near_full_rank_fraction=0.9,
        min_flip_over_random=0.05,
        sufficiency_retention_fraction=0.9,
        minimality_mode="zero",
    )
    summary = script.summarise_cell(8, 3, 16, [record])
    assert summary["n_measured"] == 1
    assert summary["used_blocks"] == list(used_blocks)
    assert sum(summary["verdict_counts"].values()) == 1
    for mode in ABLATION_MODES:
        assert mode in summary["completeness_accuracy_retention_fraction"]
    json.dumps(summary)


def test_cli_measure_writes_pending_when_i15_is_missing(dihedral_run, tmp_path):
    script = _load_script()
    out_dir = tmp_path / "used_set_audit"
    runs_root = dihedral_run.parent
    code = script.main(
        [
            "measure",
            "--runs-dir",
            str(runs_root),
            "--out-dir",
            str(out_dir),
            "--coset-dir",
            str(tmp_path / "no-coset-dir"),
            "--threshold",
            "0.0",
        ]
    )
    assert code == 0  # a pending cell was never attempted, so never counts as a skip
    payload = json.loads((out_dir / "used_set_audit_8_3_w16.json").read_text())
    assert payload["status"] == "pending"
    assert len(payload["missing"]) == 1


def test_cli_measure_writes_one_json_per_cell_when_i15_is_present(
    dihedral_run, iso_ablation_dir, tmp_path
):
    script = _load_script()
    out_dir = tmp_path / "used_set_audit"
    runs_root = dihedral_run.parent
    code = script.main(
        [
            "measure",
            "--runs-dir",
            str(runs_root),
            "--out-dir",
            str(out_dir),
            "--coset-dir",
            str(iso_ablation_dir),
            "--threshold",
            "0.0",
            "--n-random",
            "2",
        ]
    )
    assert code == 0
    payload = json.loads((out_dir / "used_set_audit_8_3_w16.json").read_text())
    assert payload["status"] == "measured"
    assert payload["summary"]["n_measured"] == 1
    assert len(payload["runs"]) == 1


def test_cli_measure_exits_nonzero_when_a_run_is_skipped(dihedral_run, iso_ablation_dir, tmp_path):
    script = _load_script()
    out_dir = tmp_path / "used_set_audit_skip"
    runs_root = dihedral_run.parent
    code = script.main(
        [
            "measure",
            "--runs-dir",
            str(runs_root),
            "--out-dir",
            str(out_dir),
            "--coset-dir",
            str(iso_ablation_dir),
            "--threshold",
            "0.99",
            "--n-random",
            "2",
        ]
    )
    assert code == 1
    payload = json.loads((out_dir / "used_set_audit_8_3_w16.json").read_text())
    assert payload["runs"][0]["status"] == "skipped"
