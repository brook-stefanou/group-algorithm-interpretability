"""The I-20 coset-quotient-route entry point (scripts/measure_coset_quotient.py).

Runs the real ``measure`` command against short training runs on the fixture
corpus -- offline, no W&B, no network. The script is an executable entry
point, not part of the installed package, so it is loaded by file path
(mirrors ``tests/test_measure_gcr_readout.py``). D8 (order 8, index 3) has
several normal subgroups (a usable cell); C7 (order 7, index 1, prime) has
none (an unusable, screened-out cell).
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

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "measure_coset_quotient.py"

EPOCHS = 10


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("measure_coset_quotient", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["measure_coset_quotient"] = module
    spec.loader.exec_module(module)
    return module


def _train(runs_root: Path, order: int, index: int, *, width: int = 16, seed: int = 0) -> Path:
    config = ProjectConfig(
        device="cpu",
        seed=seed,
        data={"group": {"order": order, "index": index}, "train_frac": 0.5, "split_seed": 0},
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
        experiment=ExperimentConfig(name="coset-quotient-measure"),
    )
    experiment = GroupGeneralizationExperiment(config, runs_root=runs_root)
    experiment.execute()
    return experiment.run_dir


@pytest.fixture(scope="module")
def dihedral_run(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("coset-quotient-runs")
    return _train(root, 8, 3)


@pytest.fixture(scope="module")
def c7_run(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("coset-quotient-c7-runs")
    return _train(root, 7, 1)


def test_screen_cell_reports_usable_for_a_group_with_a_normal_subgroup():
    script = _load_script()
    screen = script.screen_cell(8, 3)
    assert screen["status"] == "screened"
    assert screen["usable"] is True
    assert screen["n_quotients"] > 0


def test_screen_cell_reports_unusable_for_a_prime_order_group():
    script = _load_script()
    screen = script.screen_cell(7, 1)
    assert screen["status"] == "screened"
    assert screen["usable"] is False
    assert screen["n_quotients"] == 0


def test_coset_quotient_run_produces_a_full_record(dihedral_run):
    script = _load_script()
    record = script.coset_quotient_run(dihedral_run, threshold=0.0, n_random=2)
    assert record["status"] == "measured"
    assert record["group"] == {"order": 8, "index": 3, "name": "SmallGroup(8,3)"}

    route = record["coset_quotient_route"]
    assert route["defined"] is True
    assert route["n_quotients"] > 0
    for probe in route["quotients"]:
        assert "sufficiency" in probe
        assert "necessity" in probe
    assert "discrimination_note" in route

    provenance = record["provenance"]
    assert provenance["config_hash"]
    assert provenance["checkpoint_sha256"]
    assert provenance["instrument_code_sha256"]
    assert provenance["group_artifact_path"].endswith("smallgroup_8_3.npz")
    json.dumps(record)  # the whole record must be JSON-serialisable


def test_coset_quotient_run_skips_a_run_with_no_stable_checkpoint(dihedral_run):
    script = _load_script()
    record = script.coset_quotient_run(dihedral_run, threshold=0.99, n_random=2)
    assert record["status"] == "skipped"
    assert record["checkpoint_selection"]["reason"] is not None
    assert "coset_quotient_route" not in record


def test_discover_cells_groups_by_order_index_width(tmp_path):
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
    _make_run("c_incomplete", order=8, index=3, width=128, seed=2, completed=False)

    cells = script.discover_cells(runs_dir)
    assert set(cells.keys()) == {(8, 3, 128)}
    assert [p.name for p in cells[(8, 3, 128)]] == ["a_seed0", "b_seed1"]


def test_summarise_cell_aggregates_per_quotient(dihedral_run):
    script = _load_script()
    record = script.coset_quotient_run(dihedral_run, threshold=0.0, n_random=2)
    summary = script.summarise_cell(8, 3, 16, [record])
    assert summary["n_measured"] == 1
    route = record["coset_quotient_route"]
    assert len(summary["quotients"]) == route["n_quotients"]
    for entry in summary["quotients"]:
        assert entry["coset_accuracy_over_random"]["n"] == 1
        assert entry["coset_accuracy_drop_over_random"]["n"] == 1
    assert sum(q["n_decisive"] for q in summary["quotients"]) == 1
    json.dumps(summary)


def test_cli_measure_writes_one_json_per_usable_cell(dihedral_run, tmp_path):
    script = _load_script()
    out_dir = tmp_path / "coset_quotient"
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
            "--n-random",
            "2",
        ]
    )
    assert code == 0
    out_path = out_dir / "coset_quotient_8_3_w16.json"
    assert out_path.is_file()
    payload = json.loads(out_path.read_text())
    assert payload["cell"] == {"order": 8, "index": 3, "width": 16}
    assert payload["summary"]["n_measured"] == 1
    assert len(payload["runs"]) == 1


def test_cli_measure_skips_an_unusable_cell_and_writes_nothing(c7_run, tmp_path):
    script = _load_script()
    out_dir = tmp_path / "coset_quotient_c7"
    runs_root = c7_run.parent
    code = script.main(
        [
            "measure",
            "--runs-dir",
            str(runs_root),
            "--out-dir",
            str(out_dir),
            "--threshold",
            "0.0",
            "--n-random",
            "2",
        ]
    )
    assert code == 0  # never attempted, so never counts as a skip
    assert not out_dir.exists() or list(out_dir.iterdir()) == []


def test_cli_measure_exits_nonzero_when_a_run_is_skipped(dihedral_run, tmp_path):
    script = _load_script()
    out_dir = tmp_path / "coset_quotient_skip"
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
            "--n-random",
            "2",
        ]
    )
    assert code == 1
    payload = json.loads((out_dir / "coset_quotient_8_3_w16.json").read_text())
    assert payload["runs"][0]["status"] == "skipped"
