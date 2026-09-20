"""The GCR matrix-product entry point (scripts/measure_gcr_matmul.py).

Runs the real ``measure`` command against real short training runs on the
fixture corpus -- offline, no W&B, no network. The script is an executable
entry point, not part of the installed package, so it is loaded by file path
(mirrors ``tests/test_measure_occupancy.py``).

The synthetic test corpus (``tests/support_artifacts.py``, out of this
driver's scope to extend) carries only cyclic groups plus S3/D8/Q8 -- no
group with an irrep of degree >= 3 -- so this smoke test uses D8
(SmallGroup(8,3), max irrep degree 2) and exercises the driver's ``low_power``
branch: checkpoint selection, provenance, per-cell aggregation and JSON I/O.
The instrument's own decisive-branch (degree >= 3) correctness is covered by
``tests/test_gcr_matmul.py`` against synthetic irrep matrices, not here.
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

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "measure_gcr_matmul.py"

EPOCHS = 2


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("measure_gcr_matmul", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["measure_gcr_matmul"] = module
    spec.loader.exec_module(module)
    return module


def _train(runs_root: Path, seed: int, group: str = "D8") -> Path:
    config = ProjectConfig(
        device="cpu",
        seed=seed,
        data={"group": group, "train_frac": 0.5, "split_seed": 0},
        model={"d_model": 16, "d_mlp": 32, "n_heads": 1},
        optim={"epochs": EPOCHS, "log_every": 1, "print_every": 1},
        logging=LoggingConfig(mode="disabled"),
        experiment=ExperimentConfig(name="gcr-matmul-measure"),
    )
    experiment = GroupGeneralizationExperiment(config, runs_root=runs_root)
    experiment.execute()
    return experiment.run_dir


@pytest.fixture(scope="module")
def two_seed_runs(tmp_path_factory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("gcr-matmul-runs")
    return _train(root, seed=0), _train(root, seed=1)


# ---------------------------------------------------------------------------
# measure_gcr_matmul_run (library)
# ---------------------------------------------------------------------------


def test_measure_run_produces_a_measured_record_with_provenance(two_seed_runs):
    """D8 at threshold 0.0 (a 2-epoch model never clears 0.99) always has a
    stable checkpoint, so the record is measured with its one degree-2 irrep
    fitted (flagged low_power) and the full provenance block."""
    script = _load_script()
    run_dir, _ = two_seed_runs
    record = script.measure_gcr_matmul_run(run_dir, threshold=0.0)
    assert record["status"] == "measured"
    assert record["group"] == {"order": 8, "index": 3, "name": "SmallGroup(8,3)"}

    provenance = record["provenance"]
    assert provenance["config_hash"]
    assert provenance["config_group_hash"]
    assert provenance["checkpoint_sha256"]
    assert provenance["group_artifact_sha256"]
    assert "instrument_code_sha256" in provenance
    assert "gcr_matmul.py" in provenance["instrument_code_sha256"]

    result = record["gcr_matmul"]
    assert result["status"] == "low_power"
    assert result["max_irrep_degree"] == 2
    assert result["low_power"] is True
    assert len(result["fits"]) >= 1
    for fit in result["fits"]:
        assert fit["irrep_degree"] >= 2
        assert fit["low_power"] is True
        assert set(fit) >= {
            "block_index",
            "mp_fve_heldout",
            "bilinear_fve_heldout",
            "ab_fve_heldout",
            "mp_fraction_of_ab",
            "mp_minus_bilinear_heldout",
            "mp_favoured",
            "credits_gcr",
            "low_power",
        }
    json.dumps(record)  # the whole record must be JSON-serialisable


def test_measure_run_skips_a_run_with_no_stable_checkpoint(two_seed_runs):
    run_dir, _ = two_seed_runs
    record = script_module().measure_gcr_matmul_run(run_dir, threshold=0.99)
    assert record["status"] == "skipped"
    assert record["checkpoint_selection"]["reason"] is not None
    assert "gcr_matmul" not in record


def script_module() -> ModuleType:
    return _load_script()


# ---------------------------------------------------------------------------
# Cell aggregation
# ---------------------------------------------------------------------------


def test_aggregate_cell_summarises_every_occupied_irrep_across_seeds(two_seed_runs):
    script = _load_script()
    records = [script.measure_gcr_matmul_run(run_dir, threshold=0.0) for run_dir in two_seed_runs]
    aggregate = script._aggregate_cell(records)
    assert aggregate["n_seeds_measured"] == 2
    assert aggregate["n_seeds_skipped"] == 0
    assert aggregate["seeds_measured"] == [0, 1]
    assert len(aggregate["irreps"]) >= 1
    for irrep in aggregate["irreps"]:
        assert irrep["n_seeds"] == 2
        assert irrep["mp_fve_heldout"]["n"] == 2
        assert irrep["mp_fve_heldout"]["bootstrap_ci_95"] is not None
        assert 0.0 <= irrep["mp_favoured_fraction"] <= 1.0
        assert 0.0 <= irrep["credits_gcr_fraction"] <= 1.0
    json.dumps(aggregate)


def test_aggregate_cell_reports_a_single_seed_with_no_bootstrap_ci(two_seed_runs):
    script = _load_script()
    run_dir, _ = two_seed_runs
    record = script.measure_gcr_matmul_run(run_dir, threshold=0.0)
    aggregate = script._aggregate_cell([record])
    assert aggregate["n_seeds_measured"] == 1
    for irrep in aggregate["irreps"]:
        assert irrep["mp_fve_heldout"]["bootstrap_ci_95"] is None


def test_cell_key_separates_by_group_and_width(two_seed_runs):
    script = _load_script()
    run_dir, _ = two_seed_runs
    record = script.measure_gcr_matmul_run(run_dir, threshold=0.0)
    assert script._cell_key(record) == (8, 3, 16)


# ---------------------------------------------------------------------------
# The CLI entry point
# ---------------------------------------------------------------------------


def test_cli_measure_writes_one_file_per_cell(two_seed_runs, tmp_path, capsys):
    script = _load_script()
    out_dir = tmp_path / "gcr_matmul"
    code = script.main(
        [
            "measure",
            str(two_seed_runs[0]),
            str(two_seed_runs[1]),
            "--threshold",
            "0.0",
            "--out-dir",
            str(out_dir),
        ]
    )
    assert code == 0
    out_path = out_dir / "gcr_matmul_8_3_w16.json"
    assert out_path.is_file()
    payload = json.loads(out_path.read_text())
    assert payload["group"] == {"order": 8, "index": 3, "name": "SmallGroup(8,3)"}
    assert len(payload["runs"]) == 2
    assert payload["aggregate"]["n_seeds_measured"] == 2
    printed = capsys.readouterr().out
    assert "credits_gcr" in printed


def test_cli_measure_exits_nonzero_when_a_run_is_skipped(two_seed_runs, tmp_path):
    script = _load_script()
    out_dir = tmp_path / "gcr_matmul_skip"
    code = script.main(
        [
            "measure",
            str(two_seed_runs[0]),
            "--threshold",
            "0.99",
            "--out-dir",
            str(out_dir),
        ]
    )
    assert code == 1
    payload = json.loads((out_dir / "gcr_matmul_8_3_w16.json").read_text())
    assert payload["runs"][0]["status"] == "skipped"
    assert payload["aggregate"]["n_seeds_measured"] == 0


def test_write_json_sanitises_nonfinite_to_null(tmp_path):
    script = _load_script()
    out = tmp_path / "nan.json"
    script._write_json(out, {"value": float("nan"), "finite": 1.5})
    text = out.read_text()
    assert "NaN" not in text

    def _reject(token: str) -> float:
        raise ValueError(f"non-RFC constant token in output: {token}")

    loaded = json.loads(text, parse_constant=_reject)
    assert loaded["value"] is None
    assert loaded["finite"] == 1.5
