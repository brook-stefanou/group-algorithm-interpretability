"""The mechanism-probe entry point (scripts/measure_probes.py).

Runs the real ``measure`` command against short training runs on the fixture
corpus -- offline, no W&B, no network. The script is an executable entry point,
not part of the installed package, so it is loaded by file path (mirrors
``tests/test_measure_occupancy.py``). The decisive end-to-end property: a D8 run
yields a measured signed-cyclic record, a Q8 run yields ``UNDEFINED`` for it.
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

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "measure_probes.py"

EPOCHS = 10


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("measure_probes", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["measure_probes"] = module
    spec.loader.exec_module(module)
    return module


def _train(runs_root: Path, name: str) -> Path:
    config = ProjectConfig(
        device="cpu",
        seed=0,
        data={"group": name, "train_frac": 0.5, "split_seed": 0},
        model={"d_model": 16, "d_mlp": 32, "n_heads": 1},
        optim={"epochs": EPOCHS, "log_every": 1, "print_every": 1},
        snapshot={
            "enabled": True,
            "interval": 4,
            "log_dense_until": 4,
            "event_based": False,
            "final_window_epochs": 3,
        },
        logging=LoggingConfig(mode="disabled"),
        experiment=ExperimentConfig(name="probe-measure"),
    )
    experiment = GroupGeneralizationExperiment(config, runs_root=runs_root)
    experiment.execute()
    return experiment.run_dir


@pytest.fixture(scope="module")
def dihedral_and_quaternion_runs(tmp_path_factory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("probe-runs")
    return _train(root, "D8"), _train(root, "Q8")


def test_probe_run_produces_a_full_record_on_the_dihedral_member(dihedral_and_quaternion_runs):
    d8_run, _ = dihedral_and_quaternion_runs
    script = _load_script()
    record = script.probe_run(d8_run, threshold=0.0)
    assert record["status"] == "measured"
    assert record["group"] == {"order": 8, "index": 3, "name": "SmallGroup(8,3)"}

    probes = record["probes"]
    # I-28: the power-map probe reports a chance-corrected metric.
    assert probes["power_map"]["metric"] == "adjusted_balanced_accuracy"
    # I-27: the signed-cyclic structure exists on the dihedral member.
    assert probes["signed_cyclic"]["status"] == "measured"
    assert probes["signed_cyclic"]["action_sign"] == -1
    # I-28b + I-26 present.
    assert "involution_ablation" in probes
    assert probes["carry_digit"]["status"] in ("measured", "UNDEFINED")

    provenance = record["provenance"]
    assert provenance["config_hash"]
    assert provenance["checkpoint_sha256"]
    assert provenance["instrument_code_sha256"]
    json.dumps(record)  # the whole record must be JSON-serialisable


def test_probe_run_is_undefined_for_signed_cyclic_on_the_quaternionic_member(
    dihedral_and_quaternion_runs,
):
    """DECISIVE, end to end: the trained Q8 run carries no signed-cyclic
    coordinate system, so the instrument reports UNDEFINED through the whole
    pipeline -- not a low score."""
    _, q8_run = dihedral_and_quaternion_runs
    script = _load_script()
    record = script.probe_run(q8_run, threshold=0.0)
    assert record["status"] == "measured"
    assert record["probes"]["signed_cyclic"]["status"] == "UNDEFINED"
    # Q8's single involution makes the involution *probe* degenerate (singleton
    # class -> UNDEFINED), while the ablation direction is still defined.
    assert record["probes"]["power_map"]["probes"]["is_involution"] == "UNDEFINED"
    assert record["probes"]["involution_ablation"]["status"] == "measured"


def test_probe_run_skips_a_run_with_no_stable_checkpoint(dihedral_and_quaternion_runs):
    d8_run, _ = dihedral_and_quaternion_runs
    script = _load_script()
    record = script.probe_run(d8_run, threshold=0.99)
    assert record["status"] == "skipped"
    assert record["checkpoint_selection"]["reason"] is not None
    assert "probes" not in record


def test_cli_measure_writes_per_run_and_collected_output(dihedral_and_quaternion_runs, tmp_path):
    script = _load_script()
    d8_run, q8_run = dihedral_and_quaternion_runs
    out = tmp_path / "probes.json"
    code = script.main(
        ["measure", str(d8_run), str(q8_run), "--threshold", "0.0", "--out", str(out)]
    )
    assert code == 0
    payload = json.loads(out.read_text())
    assert len(payload["runs"]) == 2
    for run_dir in (d8_run, q8_run):
        per_run = json.loads((run_dir / "analysis" / "probes.json").read_text())
        assert per_run["status"] == "measured"


def test_cli_measure_exits_nonzero_when_a_run_is_skipped(dihedral_and_quaternion_runs, tmp_path):
    script = _load_script()
    d8_run, _ = dihedral_and_quaternion_runs
    out = tmp_path / "probes-skip.json"
    code = script.main(["measure", str(d8_run), "--threshold", "0.99", "--out", str(out)])
    assert code == 1
    payload = json.loads(out.read_text())
    assert payload["runs"][0]["status"] == "skipped"
