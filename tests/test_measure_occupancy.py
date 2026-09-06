"""The occupancy entry point (scripts/measure_occupancy.py + instruments/report.py).

Runs the real `measure` and `null-gate` commands against real short training
runs on the fixture corpus -- offline, no W&B, no network. The script is an
executable entry point, not part of the installed package, so it is loaded by
file path (mirrors ``tests/test_run_batch_command.py``).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from group_algorithm_interp.config import ExperimentConfig, LoggingConfig, ProjectConfig
from group_algorithm_interp.experiment import GroupGeneralizationExperiment
from group_algorithm_interp.instruments.occupancy import (
    dirichlet_noise_floor,
    restrict_to_nontrivial,
    total_variation,
)
from group_algorithm_interp.instruments.report import (
    instrument_code_hashes,
    measure_run,
    null_gate,
    pool_records,
)

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "measure_occupancy.py"

EPOCHS = 10


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("measure_occupancy", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["measure_occupancy"] = module
    spec.loader.exec_module(module)
    return module


def _train(runs_root: Path, seed: int, group: str = "C4") -> Path:
    config = ProjectConfig(
        device="cpu",
        seed=seed,
        data={"group": group, "train_frac": 0.5, "split_seed": 0},
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
        experiment=ExperimentConfig(name="occ-measure"),
    )
    experiment = GroupGeneralizationExperiment(config, runs_root=runs_root)
    experiment.execute()
    return experiment.run_dir


@pytest.fixture(scope="module")
def two_seed_runs(tmp_path_factory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("occ-runs")
    return _train(root, seed=0), _train(root, seed=1)


@pytest.fixture(scope="module")
def two_seed_d8_runs(tmp_path_factory) -> tuple[Path, Path]:
    """Two seeds of a non-abelian group (D8) so the coset arm is *defined* --
    the template-comparison branch of ``pool_records`` runs on a list of
    templates rather than the abelian ``UNDEFINED`` string."""
    root = tmp_path_factory.mktemp("occ-d8-runs")
    return _train(root, seed=0, group="D8"), _train(root, seed=1, group="D8")


# ---------------------------------------------------------------------------
# measure_run / pool_records (library)
# ---------------------------------------------------------------------------


def test_measure_run_produces_a_full_record(two_seed_runs):
    """A 10-epoch model never clears 0.99, so the threshold is dropped to 0 --
    the bar is an explicit recorded parameter, which is what makes short-run
    testing possible at all."""
    run_dir, _ = two_seed_runs
    record = measure_run(run_dir, threshold=0.0)
    assert record["status"] == "measured"
    assert record["group"] == {"order": 4, "index": 1, "name": "SmallGroup(4,1)"}
    assert record["n_units"] == 32

    selection = record["checkpoint_selection"]
    assert selection["rule"] == "final_window"
    assert selection["checkpoint"] == f"final_epoch_{EPOCHS - 1}.pt"
    assert selection["threshold"] == 0.0
    assert selection["substitution"] is None

    # Provenance: every hash the study's contract requires.
    provenance = record["provenance"]
    assert provenance["config_hash"]
    assert provenance["config_group_hash"]
    assert provenance["dataset_spec_hash"]
    assert provenance["checkpoint_sha256"]
    assert set(provenance["instrument_code_sha256"]) == set(instrument_code_hashes())

    for argument in ("left", "right"):
        block = record["occupancy"][argument]
        assert sum(block["occupancy"]) == pytest.approx(1.0)
        assert sum(block["nontrivial"]["occupancy"]) == pytest.approx(1.0)
        assert block["full"]["noise_floor"] > 0
        assert block["nontrivial"]["tv_over_floor"] == pytest.approx(
            block["nontrivial"]["tv_to_null"] / block["nontrivial"]["noise_floor"]
        )
        assert len(block["per_neuron_top_share"]) == 32
    # C4 is abelian: the coset arm is structurally UNDEFINED, not a number.
    assert record["templates"]["entries"] == "UNDEFINED"
    for argument in ("left", "right"):
        assert record["occupancy"][argument]["template_comparisons"] == "UNDEFINED"
    json.dumps(record)  # the whole record must be JSON-serialisable


def test_measure_run_skips_a_run_with_no_stable_checkpoint(two_seed_runs):
    """At the real 0.99 bar this short run has no stable epoch anywhere: the
    record says skipped, keeps the full selection record (reported as data),
    and carries no occupancy numbers."""
    run_dir, _ = two_seed_runs
    record = measure_run(run_dir, threshold=0.99)
    assert record["status"] == "skipped"
    assert record["checkpoint_selection"]["reason"] is not None
    assert "occupancy" not in record


def test_pool_records_pools_seeds_within_one_config(two_seed_runs):
    records = [measure_run(run_dir, threshold=0.0) for run_dir in two_seed_runs]
    pooled = pool_records(records)
    assert len(pooled) == 1  # same experiment config apart from the seed
    entry = pooled[0]
    assert entry["n_runs"] == 2
    assert entry["seeds"] == [0, 1]
    assert entry["n_units"] == 64  # 2 seeds x 32 neurons: the floor pools
    assert entry["config_group_hash"] == records[0]["provenance"]["config_group_hash"]
    for argument in ("left", "right"):
        block = entry["occupancy"][argument]
        assert sum(block["occupancy"]) == pytest.approx(1.0)
        assert block["full"]["noise_floor"] == pytest.approx(
            records[0]["occupancy"][argument]["full"]["noise_floor"] / (2**0.5)
        )
    json.dumps(pooled)


def test_pool_records_ignores_skipped_runs(two_seed_runs):
    run_dir, other = two_seed_runs
    records = [
        measure_run(run_dir, threshold=0.0),
        measure_run(other, threshold=0.99),  # skipped
    ]
    pooled = pool_records(records)
    assert len(pooled) == 1
    assert pooled[0]["n_runs"] == 1


def test_measure_run_floor_uses_nonzero_energy_units(two_seed_runs):
    """The Dirichlet floor's N is the nonzero-energy neuron count, not the raw
    d_mlp -- both counts stay in the record so either is recoverable."""
    run_dir, _ = two_seed_runs
    record = measure_run(run_dir, threshold=0.0)
    pi0 = np.asarray(record["analytic_null"])
    for argument in ("left", "right"):
        block = record["occupancy"][argument]
        assert block["n_units"] == 32
        assert block["n_nonzero_energy_units"] == block["n_units"] - block["n_zero_energy_units"]
        assert block["full"]["noise_floor"] == pytest.approx(
            dirichlet_noise_floor(pi0, block["n_nonzero_energy_units"])
        )


def test_pool_records_floor_excludes_dead_neurons(two_seed_runs):
    """Finding 10: the pooled floor's N sums the members' nonzero-energy counts,
    so simulated dead neurons shrink N (and raise the floor) rather than being
    counted as signal-bearing units."""
    records = [measure_run(run_dir, threshold=0.0) for run_dir in two_seed_runs]
    for record in records:
        for argument in ("left", "right"):
            record["occupancy"][argument]["n_nonzero_energy_units"] = 10
    pooled = pool_records(records)[0]
    pi0 = np.asarray(pooled["analytic_null"])
    for argument in ("left", "right"):
        block = pooled["occupancy"][argument]
        assert block["n_nonzero_energy_units"] == 20
        assert block["full"]["noise_floor"] == pytest.approx(dirichlet_noise_floor(pi0, 20))


def test_pool_records_rejects_a_duplicate_run_id(two_seed_runs):
    """Passing one record twice doubles N and shrinks the floor by 1/sqrt(2);
    it must raise naming the offending run rather than pool it silently."""
    run_dir, _ = two_seed_runs
    record = measure_run(run_dir, threshold=0.0)
    with pytest.raises(ValueError, match="duplicate run_id"):
        pool_records([record, record])


def test_pool_records_rejects_a_duplicate_seed_within_a_pool(two_seed_runs):
    """Two distinct run_ids sharing a (config_group_hash, seed) are same-seed
    reruns -- the corpus has these -- and must raise, not double the floor's N."""
    run_dir, _ = two_seed_runs
    first = measure_run(run_dir, threshold=0.0)
    rerun = measure_run(run_dir, threshold=0.0)  # same seed + config_group_hash
    rerun["run_id"] = f"{first['run_id']}-rerun"  # a distinct id
    with pytest.raises(ValueError, match="duplicate seed"):
        pool_records([first, rerun])


def test_pool_records_carries_a_provenance_block(two_seed_runs):
    """Finding 3: pooled records gain the analysis-side provenance plus the
    members' shared checkpoint metric/threshold and per-member epochs."""
    records = [measure_run(run_dir, threshold=0.0) for run_dir in two_seed_runs]
    provenance = pool_records(records)[0]["provenance"]
    assert set(provenance) >= {
        "analysis_git_commit",
        "analysis_git_dirty",
        "analysed_at",
        "instrument_code_sha256",
        "member_metric",
        "member_threshold",
        "member_checkpoint_epochs",
    }
    assert provenance["member_metric"] == "val/accuracy"
    assert provenance["member_threshold"] == 0.0
    assert len(provenance["member_checkpoint_epochs"]) == 2
    assert set(provenance["instrument_code_sha256"]) == set(instrument_code_hashes())


def test_pool_records_rejects_members_at_mixed_thresholds(two_seed_runs):
    run_a, run_b = two_seed_runs
    a = measure_run(run_a, threshold=0.0)
    b = measure_run(run_b, threshold=0.0)
    b["checkpoint_selection"]["threshold"] = 0.5  # a different bar within the pool
    with pytest.raises(ValueError, match="mixes checkpoint metrics/thresholds"):
        pool_records([a, b])


def test_pool_records_flags_and_warns_on_a_missing_config_group_hash(two_seed_runs, capsys):
    """Finding 4: a manifest with no config_group_hash pools alone under the
    run_id fallback, but loudly -- a flag on the record and a warning."""
    run_dir, _ = two_seed_runs
    record = measure_run(run_dir, threshold=0.0)
    record["provenance"]["config_group_hash"] = None
    pooled = pool_records([record])[0]
    assert pooled["pool_key_fallback"] is True
    assert pooled["config_group_hash"] == record["run_id"]
    assert "pool_key_fallback" in capsys.readouterr().err


def test_pool_records_is_order_independent(two_seed_runs):
    """Finding 5: the same record set in either order yields byte-identical
    pooled output, modulo the intentional analysed_at timestamp."""
    records = [measure_run(run_dir, threshold=0.0) for run_dir in two_seed_runs]

    def _strip_timestamp(pooled):
        for record in pooled:
            record["provenance"].pop("analysed_at", None)
        return pooled

    forward = _strip_timestamp(pool_records(list(records)))
    reverse = _strip_timestamp(pool_records(list(reversed(records))))
    assert forward == reverse


def test_pool_records_exercises_the_template_comparison_branch(two_seed_d8_runs):
    """The non-abelian (coset-defined) pooling branch -- untested by any abelian
    fixture. D8's template comparisons are a list, and the pooled TVs match a
    recomputation from the pooled occupancy against each template."""
    records = [measure_run(run_dir, threshold=0.0) for run_dir in two_seed_d8_runs]
    pooled = pool_records(records)[0]
    entries = pooled["templates"]["entries"]
    assert entries != "UNDEFINED"
    trivial = pooled["trivial_block_index"]
    for argument in ("left", "right"):
        block = pooled["occupancy"][argument]
        comparisons = block["template_comparisons"]
        assert isinstance(comparisons, list)
        assert len(comparisons) == len(entries)
        occupancy = np.asarray(block["occupancy"], dtype=np.float64)
        occupancy_nt = restrict_to_nontrivial(occupancy, trivial)
        for comparison, entry in zip(comparisons, entries, strict=True):
            template = np.asarray(entry["template"], dtype=np.float64)
            assert comparison["tv_occupancy_to_template"] == pytest.approx(
                total_variation(occupancy, template)
            )
            assert comparison["tv_nontrivial_occupancy_to_template"] == pytest.approx(
                total_variation(occupancy_nt, restrict_to_nontrivial(template, trivial))
            )


# ---------------------------------------------------------------------------
# null_gate (I-11, library)
# ---------------------------------------------------------------------------


def test_null_gate_reports_both_conditions_without_a_verdict():
    """D8/Q8 is a genuine CT-equal contrast pair in the fixture corpus: shared
    degrees, shared block ranks, so the free pass must be asserted
    (pi0_equal_as_multisets True), and the per-seed paired difference carries
    a bootstrap CI rather than any verdict label."""
    record = null_gate(
        (8, 3), (8, 4), [0, 1, 2], model_config={"d_model": 16, "d_mlp": 32, "n_heads": 1}
    )
    assert record["pi0_equal_as_multisets"] is True
    assert record["seeds"] == [0, 1, 2]
    for label in ("a", "b"):
        condition = record["conditions"][label]
        for argument in ("left", "right"):
            for form in ("full", "nontrivial"):
                stats_block = condition["tv_to_own_null"][argument][form]
                assert len(stats_block["tv_per_seed"]) == 3
                assert all(v >= 0.0 for v in stats_block["tv_per_seed"])
    difference = record["paired_difference_a_minus_b"]["left"]["nontrivial"]
    low, high = difference["bootstrap_ci_95"]
    assert low <= difference["mean"] <= high
    assert "verdict" not in record
    json.dumps(record)


def test_null_gate_records_the_full_model_config():
    """Finding 2: the gate records the whole model configuration build_model
    consumed, so a run at the wrong width/arch cannot be filed as a campaign's
    calibration unfalsifiably."""
    record = null_gate(
        (8, 3), (8, 4), [0, 1], model_config={"d_model": 16, "d_mlp": 32, "n_heads": 1}
    )
    assert record["model"] == {
        "arch": "transformer",
        "d_model": 16,
        "n_heads": 1,
        "activation": "relu",
        "d_mlp": 32,
    }


def test_null_gate_requires_at_least_two_seeds():
    """Finding 6: a single-seed gate is rejected up front with a clear message,
    not deep inside stats.bootstrap_ci."""
    with pytest.raises(ValueError, match="at least 2 seeds"):
        null_gate((8, 3), (8, 4), [0], model_config={"d_model": 16, "d_mlp": 32, "n_heads": 1})


# ---------------------------------------------------------------------------
# The CLI entry point
# ---------------------------------------------------------------------------


def test_cli_measure_writes_per_run_and_pooled_output(two_seed_runs, tmp_path, capsys):
    script = _load_script()
    out = tmp_path / "occupancy.json"
    code = script.main(
        [
            "measure",
            str(two_seed_runs[0]),
            str(two_seed_runs[1]),
            "--threshold",
            "0.0",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    payload = json.loads(out.read_text())
    assert len(payload["runs"]) == 2
    assert len(payload["pooled"]) == 1
    for run_dir in two_seed_runs:
        per_run = json.loads((run_dir / "analysis" / "occupancy.json").read_text())
        assert per_run["status"] == "measured"
    printed = capsys.readouterr().out
    assert "TV(left, nontrivial)" in printed


def test_cli_measure_exits_nonzero_when_a_run_is_skipped(two_seed_runs, tmp_path):
    script = _load_script()
    out = tmp_path / "occupancy-skip.json"
    code = script.main(["measure", str(two_seed_runs[0]), "--threshold", "0.99", "--out", str(out)])
    assert code == 1
    payload = json.loads(out.read_text())
    assert payload["runs"][0]["status"] == "skipped"
    assert payload["pooled"] == []


def test_cli_null_gate_writes_the_record(tmp_path):
    script = _load_script()
    out = tmp_path / "gate.json"
    code = script.main(
        [
            "null-gate",
            "--group-a",
            "8,3",
            "--group-b",
            "8,4",
            "--seeds",
            "0:2",
            "--d-model",
            "16",
            "--d-mlp",
            "32",
            "--n-heads",
            "1",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    record = json.loads(out.read_text())
    assert record["instrument"] == "occupancy-null-calibration"
    assert record["seeds"] == [0, 1]


def test_write_json_sanitises_nonfinite_to_null(tmp_path):
    """Finding 8: non-finite floats are written as ``null`` (RFC-8259), never
    bare NaN/Infinity tokens that jq and strict parsers reject."""
    script = _load_script()
    out = tmp_path / "nan.json"
    payload = {
        "metric_value": float("nan"),
        "nested": {"pos_inf": float("inf"), "neg_inf": float("-inf")},
        "finite": 1.5,
        "mixed_list": [float("nan"), 2.0],
    }
    script._write_json(out, payload)
    text = out.read_text()
    assert "NaN" not in text
    assert "Infinity" not in text

    def _reject(token: str) -> float:
        raise ValueError(f"non-RFC constant token in output: {token}")

    loaded = json.loads(text, parse_constant=_reject)  # a strict parser must not choke
    assert loaded["metric_value"] is None
    assert loaded["nested"]["pos_inf"] is None
    assert loaded["nested"]["neg_inf"] is None
    assert loaded["finite"] == 1.5
    assert loaded["mixed_list"] == [None, 2.0]


def test_cli_seed_parsing_matches_run_batch_grammar():
    script = _load_script()
    assert script.parse_seeds("0:3") == [0, 1, 2]
    assert script.parse_seeds("0:3,7,2") == [0, 1, 2, 7]
    with pytest.raises(ValueError, match="reversed"):
        script.parse_seeds("5:1")
    with pytest.raises(ValueError, match="no seeds"):
        script.parse_seeds(",")
