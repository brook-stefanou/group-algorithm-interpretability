"""The GCR character-readout entry point (scripts/measure_gcr_readout.py).

Runs the real ``measure`` command against short training runs on the fixture
corpus -- offline, no W&B, no network. The script is an executable entry point,
not part of the installed package, so it is loaded by file path (mirrors
``tests/test_measure_probes.py``). Cell discovery/grouping is exercised
separately against synthetic run directories (no training needed there), kept
fast the same way.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from group_algorithm_interp.config import ExperimentConfig, LoggingConfig, ProjectConfig
from group_algorithm_interp.experiment import GroupGeneralizationExperiment
from group_algorithm_interp.instruments.probes import UNDEFINED

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "measure_gcr_readout.py"

EPOCHS = 10


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("measure_gcr_readout", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["measure_gcr_readout"] = module
    spec.loader.exec_module(module)
    return module


def _train(runs_root: Path, name: str, *, width: int = 16, seed: int = 0) -> Path:
    config = ProjectConfig(
        device="cpu",
        seed=seed,
        data={"group": name, "train_frac": 0.5, "split_seed": 0},
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
        experiment=ExperimentConfig(name="gcr-readout-measure"),
    )
    experiment = GroupGeneralizationExperiment(config, runs_root=runs_root)
    experiment.execute()
    return experiment.run_dir


@pytest.fixture(scope="module")
def dihedral_run(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("gcr-readout-runs")
    return _train(root, "D8")


def test_gcr_readout_run_produces_a_full_record(dihedral_run):
    script = _load_script()
    record = script.gcr_readout_run(dihedral_run, threshold=0.0)
    assert record["status"] == "measured"
    assert record["group"] == {"order": 8, "index": 3, "name": "SmallGroup(8,3)"}

    gcr = record["gcr_readout"]
    assert gcr["status"] == "measured"
    # D8's abelianisation is C2 x C2, so a Fourier-only rival exists.
    nested = gcr["primary"]["nested_comparison"]
    assert nested.get("status") != UNDEFINED
    assert "full_vs_fourier_held_out_fve_gain" in nested
    assert "n_selected" in gcr["primary"]["minimal_irrep_set"]
    assert "null" in gcr
    assert gcr["null"]["full_null_held_out_fve"] is not None

    provenance = record["provenance"]
    assert provenance["config_hash"]
    assert provenance["checkpoint_sha256"]
    assert provenance["instrument_code_sha256"]
    assert provenance["group_artifact_path"].endswith("smallgroup_8_3.npz")
    assert len(provenance["group_artifact_sha256"]) == 64  # a hex sha256 digest
    json.dumps(record)  # the whole record must be JSON-serialisable


def test_gcr_readout_run_skips_a_run_with_no_stable_checkpoint(dihedral_run):
    script = _load_script()
    record = script.gcr_readout_run(dihedral_run, threshold=0.99)
    assert record["status"] == "skipped"
    assert record["checkpoint_selection"]["reason"] is not None
    assert "gcr_readout" not in record


def test_discover_cells_groups_by_order_index_width(tmp_path):
    """Synthetic run directories (no training) to exercise the grouping and
    the "completed manifests only" filter quickly."""
    script = _load_script()
    runs_dir = tmp_path / "runs"

    def _make_run(
        name: str, *, order: int, index: int, width: int, seed: int, completed: bool
    ) -> None:
        run_dir = runs_dir / name
        run_dir.mkdir(parents=True)
        (run_dir / "manifest.yaml").write_text(
            yaml.safe_dump({"run_id": name, "status": "completed" if completed else "running"})
        )
        (run_dir / "resolved_config.yaml").write_text(
            yaml.safe_dump(
                {
                    "seed": seed,
                    "data": {"group": {"order": order, "index": index}},
                    "model": {"d_model": width},
                }
            )
        )

    _make_run("a_seed0", order=8, index=3, width=128, seed=0, completed=True)
    _make_run("b_seed1", order=8, index=3, width=128, seed=1, completed=True)
    _make_run("c_seed0_w256", order=8, index=3, width=256, seed=0, completed=True)
    _make_run("d_other_group", order=8, index=4, width=128, seed=0, completed=True)
    _make_run("e_incomplete", order=8, index=3, width=128, seed=2, completed=False)

    cells = script.discover_cells(runs_dir)
    assert set(cells.keys()) == {(8, 3, 128), (8, 3, 256), (8, 4, 128)}
    assert [p.name for p in cells[(8, 3, 128)]] == ["a_seed0", "b_seed1"]
    assert len(cells[(8, 3, 256)]) == 1
    assert len(cells[(8, 4, 128)]) == 1


def test_summarise_cell_reports_undefined_nested_comparison_when_no_seed_has_it():
    script = _load_script()
    fake_records = [
        {
            "status": "measured",
            "gcr_readout": {
                "primary": {
                    "nested_comparison": {"status": UNDEFINED, "full_held_out_fve": 0.9},
                    "minimal_irrep_set": {
                        "n_selected": 2,
                        "selected_irrep_indices": [1, 2],
                    },
                }
            },
        }
    ]
    summary = script.summarise_cell(64, 1, 128, fake_records)
    assert summary["nested_comparison"]["status"] == UNDEFINED
    assert summary["minimal_irrep_set"]["n_selected"]["mean"] == 2.0
    json.dumps(summary)


def test_cli_measure_writes_one_json_per_cell(dihedral_run, tmp_path):
    script = _load_script()
    out_dir = tmp_path / "gcr_readout"
    runs_root = dihedral_run.parent
    code = script.main(
        [
            "measure",
            "--runs-dir",
            str(runs_root),
            "--out-dir",
            str(out_dir),
            "--threshold",
            "0.0",
            "--max-seeds-per-cell",
            "50",
        ]
    )
    assert code == 0
    out_path = out_dir / "gcr_readout_8_3_w16.json"
    assert out_path.is_file()
    payload = json.loads(out_path.read_text())
    assert payload["cell"] == {"order": 8, "index": 3, "width": 16}
    assert payload["summary"]["n_measured"] == 1
    assert len(payload["runs"]) == 1


def test_cli_measure_exits_nonzero_when_a_run_is_skipped(dihedral_run, tmp_path):
    script = _load_script()
    out_dir = tmp_path / "gcr_readout_skip"
    runs_root = dihedral_run.parent
    code = script.main(
        [
            "measure",
            "--runs-dir",
            str(runs_root),
            "--out-dir",
            str(out_dir),
            "--threshold",
            "0.99",
        ]
    )
    assert code == 1
    payload = json.loads((out_dir / "gcr_readout_8_3_w16.json").read_text())
    assert payload["runs"][0]["status"] == "skipped"
