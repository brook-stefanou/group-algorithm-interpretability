"""``scripts/measure_isotypic_ablation.py``: I-15 across a cell's seeds.

Runs the real script against real short training runs on the fixture corpus
-- offline, no W&B, no network -- mirroring
``tests/test_measure_occupancy.py``. D8 (order 8, index 3) is used because it
has more than one nontrivial isotypic block, unlike the cyclic groups used
elsewhere in the occupancy tests.
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
from group_algorithm_interp.instruments.report import instrument_code_hashes

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "measure_isotypic_ablation.py"

EPOCHS = 6


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("measure_isotypic_ablation", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["measure_isotypic_ablation"] = module
    spec.loader.exec_module(module)
    return module


def _train(runs_root: Path, seed: int, group: tuple[int, int] = (8, 3)) -> Path:
    config = ProjectConfig(
        device="cpu",
        seed=seed,
        data={"group": {"order": group[0], "index": group[1]}, "train_frac": 0.8, "split_seed": 0},
        model={"d_model": 16, "d_mlp": 32, "n_heads": 1},
        optim={"epochs": EPOCHS, "log_every": 1, "print_every": 1},
        logging=LoggingConfig(mode="disabled"),
        experiment=ExperimentConfig(name="iso-ablation-measure"),
    )
    experiment = GroupGeneralizationExperiment(config, runs_root=runs_root)
    experiment.execute()
    return experiment.run_dir


@pytest.fixture(scope="module")
def three_seed_runs(tmp_path_factory) -> list[Path]:
    root = tmp_path_factory.mktemp("iso-ablation-runs")
    return [_train(root, seed=s) for s in range(3)]


# ---------------------------------------------------------------------------
# _measure_run
# ---------------------------------------------------------------------------


def test_measure_run_computes_only_isotypic_block_ablation(three_seed_runs):
    """The lean per-run path mirrors ``coset.measure_coset_run``'s
    checkpoint-selection/provenance scaffolding but never computes the
    core-free-subgroup-gated coset arm."""
    module = _load_script()
    run_dir = three_seed_runs[0]
    record = module._measure_run(run_dir, metric="val/accuracy", threshold=0.0, n_random=2)

    assert record["status"] == "measured"
    assert record["group"] == {"order": 8, "index": 3, "name": "SmallGroup(8,3)"}
    assert "isotypic_block_ablation" in record
    assert "coset_arm" not in record

    ablation = record["isotypic_block_ablation"]
    assert ablation["instrument"] == "isotypic-block-ablation"
    # D8 has more than one nontrivial isotypic block.
    assert len(ablation["blocks"]) >= 2
    for block in ablation["blocks"]:
        assert not block["is_trivial"]
        assert set(block["modes"]) == set(ABLATION_MODES)

    provenance = record["provenance"]
    assert provenance["checkpoint_sha256"]
    assert set(provenance["instrument_code_sha256"]) == set(instrument_code_hashes())
    json.dumps(record)


def test_measure_run_skips_a_run_with_no_stable_checkpoint(three_seed_runs):
    run_dir = three_seed_runs[0]
    module = _load_script()
    record = module._measure_run(run_dir, metric="val/accuracy", threshold=0.99, n_random=2)
    assert record["status"] == "skipped"
    assert record["checkpoint_selection"]["reason"] is not None
    assert "isotypic_block_ablation" not in record


# ---------------------------------------------------------------------------
# aggregate_cell
# ---------------------------------------------------------------------------


def test_aggregate_cell_pools_blocks_across_seeds(three_seed_runs):
    module = _load_script()
    records = [
        module._measure_run(run_dir, metric="val/accuracy", threshold=0.0, n_random=2)
        for run_dir in three_seed_runs
    ]
    aggregate = module.aggregate_cell(records, order=8, index=3, width=16, name="D8")

    assert aggregate["cell"] == {"order": 8, "index": 3, "width": 16, "name": "D8"}
    assert aggregate["n_seeds_requested"] == 3
    assert aggregate["n_seeds_measured"] == 3
    assert aggregate["n_seeds_skipped"] == 0
    assert aggregate["measured_seeds"] == [0, 1, 2]

    n_blocks = len(records[0]["isotypic_block_ablation"]["blocks"])
    assert len(aggregate["blocks"]) == n_blocks
    for block in aggregate["blocks"]:
        for mode in ABLATION_MODES:
            mode_entry = block["modes"][mode]
            for key in ("flip_fraction", "random_baseline", "flip_fraction_over_random"):
                summary = mode_entry[key]
                assert summary["n"] == 3
                assert summary["bootstrap_ci_95"] is not None
                assert len(summary["bootstrap_ci_95"]) == 2
                assert summary["bootstrap_ci_95"][0] <= summary["mean"] <= summary[
                    "bootstrap_ci_95"
                ][1] or (summary["bootstrap_ci_95"][0] == summary["bootstrap_ci_95"][1])
    assert len(aggregate["runs"]) == 3
    json.dumps(aggregate)


def test_aggregate_cell_excludes_skipped_seeds_from_the_aggregate(three_seed_runs):
    module = _load_script()
    measured = module._measure_run(
        three_seed_runs[0], metric="val/accuracy", threshold=0.0, n_random=2
    )
    skipped = module._measure_run(
        three_seed_runs[1], metric="val/accuracy", threshold=0.99, n_random=2
    )
    aggregate = module.aggregate_cell([measured, skipped], order=8, index=3, width=16)

    assert aggregate["n_seeds_requested"] == 2
    assert aggregate["n_seeds_measured"] == 1
    assert aggregate["n_seeds_skipped"] == 1
    assert aggregate["skipped_seeds"] == [skipped["seed"]]
    for block in aggregate["blocks"]:
        for mode in ABLATION_MODES:
            assert block["modes"][mode]["flip_fraction"]["n"] == 1
            assert block["modes"][mode]["flip_fraction"]["bootstrap_ci_95"] is None


# ---------------------------------------------------------------------------
# CLI: canonical-run resolution + explicit run_dirs + --out
# ---------------------------------------------------------------------------


def test_cli_measure_with_explicit_run_dirs_writes_aggregate(three_seed_runs, tmp_path):
    module = _load_script()
    out_path = tmp_path / "iso_ablation_8_3_w16.json"
    argv = [
        "measure",
        *[str(p) for p in three_seed_runs],
        "--order",
        "8",
        "--index",
        "3",
        "--width",
        "16",
        "--out",
        str(out_path),
        "--threshold",
        "0.0",
        "--n-random",
        "2",
    ]
    exit_code = module.main(argv)
    assert exit_code == 0
    payload = json.loads(out_path.read_text())
    assert payload["cell"]["order"] == 8
    assert payload["cell"]["index"] == 3
    assert payload["cell"]["width"] == 16
    assert payload["n_seeds_measured"] == 3


def test_cli_measure_resolves_cell_from_canonical_runs_json(three_seed_runs, tmp_path):
    module = _load_script()
    canonical_path = tmp_path / "canonical_runs.json"
    canonical_runs = {
        f"8-3-16-{EPOCHS}-seed{seed}": {
            "order": 8,
            "index": 3,
            "width": 16,
            "epochs": EPOCHS,
            "seed": seed,
            "canonical_run_id": run_dir.name,
        }
        for seed, run_dir in enumerate(three_seed_runs)
    }
    canonical_path.write_text(json.dumps({"canonical_runs": canonical_runs}))
    out_path = tmp_path / "iso_ablation_8_3_w16.json"

    argv = [
        "measure",
        "--order",
        "8",
        "--index",
        "3",
        "--width",
        "16",
        "--out",
        str(out_path),
        "--canonical-runs",
        str(canonical_path),
        "--runs-root",
        str(three_seed_runs[0].parent),
        "--max-seeds",
        "2",
        "--threshold",
        "0.0",
        "--n-random",
        "2",
    ]
    exit_code = module.main(argv)
    assert exit_code == 0
    payload = json.loads(out_path.read_text())
    # --max-seeds caps the resolved cell at 2 of the 3 available canonical seeds.
    assert payload["n_seeds_requested"] == 2
    assert payload["measured_seeds"] == [0, 1]


def test_cli_measure_exit_code_reflects_skipped_seeds(three_seed_runs, tmp_path):
    module = _load_script()
    out_path = tmp_path / "iso_ablation_8_3_w16.json"
    argv = [
        "measure",
        *[str(p) for p in three_seed_runs],
        "--order",
        "8",
        "--index",
        "3",
        "--width",
        "16",
        "--out",
        str(out_path),
        "--threshold",
        "0.99",  # unreachable in EPOCHS epochs -> every seed skipped
        "--n-random",
        "2",
    ]
    exit_code = module.main(argv)
    assert exit_code == 1
    payload = json.loads(out_path.read_text())
    assert payload["n_seeds_measured"] == 0
    assert payload["n_seeds_skipped"] == 3
