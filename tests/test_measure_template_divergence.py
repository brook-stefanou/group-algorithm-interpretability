"""The template-divergence entry point (scripts/measure_template_divergence.py).

Runs the real ``measure`` command against a short training run on a
constructed DEFINED group -- offline, no W&B, no network, no Sage/GAP. The
script is an executable entry point, not part of the installed package, so
it is loaded by file path (mirrors ``tests/test_measure_gcr_readout.py``).

No real, locally-exported panel group screens ``DEFINED`` (see
``tests/test_template_divergence.py``'s module docstring: D32/QD32/GL(2,3)
are all degenerate, Q32 has no nontrivial core-free subgroup at all), so this
test reuses that module's hand-constructed S4 (SmallGroup(24,12), H =
Stab(3), index 4, ``Ind_H^G 1`` support 10/24 -- comfortably DEFINED under
the default 0.5 threshold) as its positive case, rather than depending on a
group artifact this repo does not carry. S4's isotypic-block projectors are
placeholders in that module (unused by ``template_divergence.py``, which
reads only ``block_rank``/``irrep_degree``/``irrep_indices``), but this
driver's occupancy computation *does* need real projectors -- they gate a
model's activations, not a character table -- so this test recomputes them
here from the group's own (validated) Cayley table and character table via
the same regular-representation projector formula ``scripts/export_group.py``
uses, and serialises the result to a format-2 ``.npz`` artifact so the
driver can load it through the normal ``GROUP_ARTIFACTS_DIR`` path exactly
as it would a real exported group.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import yaml

from group_algorithm_interp.config import ExperimentConfig, LoggingConfig, ProjectConfig
from group_algorithm_interp.experiment import GroupGeneralizationExperiment
from group_algorithm_interp.groups.data import ARTIFACT_FORMAT, GroupData, IsotypicBlock

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "measure_template_divergence.py"
_S4_TEST_PATH = Path(__file__).resolve().parent / "test_template_divergence.py"

EPOCHS = 10
S4_ORDER = 24
S4_INDEX = 12


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_script() -> ModuleType:
    return _load_module("measure_template_divergence", _SCRIPT_PATH)


def _build_s4_with_real_projectors() -> GroupData:
    """``test_template_divergence.py``'s hand-constructed S4, with its
    placeholder-zero isotypic-block projectors replaced by the real ones
    computed from its own Cayley table and (orthogonality-checked)
    character table -- ``P_j = (d_j / |G|) * sum_g conj(chi_j(g)) *
    regular(g)``, exactly the formula ``scripts/export_group.py`` uses.
    Every block here is a single irrep (S4's character table is entirely
    real, so no complex-conjugate pair merging applies), so this is a
    one-irrep-per-block sum rather than the general merge."""
    fixtures = _load_module("test_template_divergence_fixtures", _S4_TEST_PATH)
    group = fixtures._build_s4()
    table = group.cayley_table
    n = group.order
    regular = np.zeros((n, n, n), dtype=np.float64)
    for g in range(n):
        regular[g, table[g], np.arange(n)] = 1.0
    new_blocks = []
    total_rank = 0
    for block in group.isotypic_blocks:
        assert len(block.irrep_indices) == 1  # S4's character table is entirely real
        irrep = group.irreps[block.irrep_indices[0]]
        character = irrep.character
        projector = (
            sum(np.conj(character[g]) * regular[g] for g in range(n)) * block.irrep_degree / n
        )
        assert np.allclose(projector.imag, 0.0, atol=1e-9)
        real_projector = projector.real
        trace = float(np.trace(real_projector))
        assert abs(trace - block.block_rank) < 1e-6
        total_rank += block.block_rank
        new_blocks.append(
            IsotypicBlock(
                projector=real_projector,
                irrep_degree=block.irrep_degree,
                block_rank=block.block_rank,
                irrep_indices=block.irrep_indices,
            )
        )
    assert total_rank == n
    return GroupData(
        order=group.order,
        index=group.index,
        description=group.description,
        element_labels=group.element_labels,
        cayley_table=group.cayley_table,
        conjugacy_classes=group.conjugacy_classes,
        character_table=group.character_table,
        frobenius_schur=group.frobenius_schur,
        irreps=group.irreps,
        isotypic_blocks=tuple(new_blocks),
        subgroups=group.subgroups,
        left_cosets=group.left_cosets,
        provenance=group.provenance,
    )


def _write_artifact(group: GroupData, path: Path) -> None:
    """Serialise ``group`` to a format-2 ``.npz`` artifact, the same shape
    ``scripts/export_group.py`` writes and ``groups/data.py::load_group``
    reads. Hand-assembled here (no Sage/GAP at test time) from a
    ``GroupData`` already fully populated in memory."""
    payload: dict[str, np.ndarray] = {
        "metadata": np.array(
            json.dumps(
                {
                    "format": ARTIFACT_FORMAT,
                    "small_group": [group.order, group.index],
                    "order": group.order,
                    "description": group.description,
                    "element_labels": list(group.element_labels),
                    "classes": len(group.conjugacy_classes),
                    "irreps": [
                        {"dimension": irrep.dimension, "field": irrep.field, "basis": irrep.basis}
                        for irrep in group.irreps
                    ],
                    "blocks": [
                        {
                            "irrep_degree": block.irrep_degree,
                            "block_rank": block.block_rank,
                            "irrep_indices": list(block.irrep_indices),
                        }
                        for block in group.isotypic_blocks
                    ],
                    "subgroups": len(group.subgroups),
                    "coset_counts": [len(cosets) for cosets in group.left_cosets],
                    "provenance": group.provenance,
                }
            )
        ),
        "cayley_table": group.cayley_table,
        "character_real": group.character_table.real,
        "character_imag": group.character_table.imag,
        "frobenius_schur": group.frobenius_schur,
    }
    for i, cls in enumerate(group.conjugacy_classes):
        payload[f"class_{i}"] = cls
    for i, irrep in enumerate(group.irreps):
        payload[f"irrep_{i}_real"] = irrep.matrices.real
        payload[f"irrep_{i}_imag"] = irrep.matrices.imag
        payload[f"irrep_{i}_character_real"] = irrep.character.real
        payload[f"irrep_{i}_character_imag"] = irrep.character.imag
    for i, block in enumerate(group.isotypic_blocks):
        payload[f"block_{i}_projector"] = block.projector
    for i, subgroup in enumerate(group.subgroups):
        payload[f"subgroup_{i}"] = subgroup
        for j, coset in enumerate(group.left_cosets[i]):
            payload[f"coset_{i}_{j}"] = coset
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


@pytest.fixture(scope="module")
def s4_artifacts_dir(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("s4-artifacts")
    _write_artifact(
        _build_s4_with_real_projectors(), directory / f"smallgroup_{S4_ORDER}_{S4_INDEX}.npz"
    )
    return directory


@pytest.fixture(scope="module")
def s4_run(tmp_path_factory, s4_artifacts_dir, monkeypatch_module) -> Path:
    root = tmp_path_factory.mktemp("template-divergence-runs")
    config = ProjectConfig(
        device="cpu",
        seed=0,
        data={"group": {"order": S4_ORDER, "index": S4_INDEX}, "train_frac": 0.5, "split_seed": 0},
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
        experiment=ExperimentConfig(name="template-divergence-measure"),
    )
    experiment = GroupGeneralizationExperiment(config, runs_root=root)
    experiment.execute()
    return experiment.run_dir


@pytest.fixture(scope="module")
def monkeypatch_module(s4_artifacts_dir):
    """``GROUP_ARTIFACTS_DIR`` must already point at the synthetic S4
    artifact before training resolves the group, so this composes as a
    dependency of ``s4_run`` rather than a plain ``monkeypatch`` (the
    function-scoped ``monkeypatch`` fixture cannot be depended on by a
    module-scoped fixture; this hand-rolled equivalent keeps the env var set
    for the whole training call and restores it afterwards)."""
    import os

    old = os.environ.get("GROUP_ARTIFACTS_DIR")
    os.environ["GROUP_ARTIFACTS_DIR"] = str(s4_artifacts_dir)
    yield
    if old is None:
        os.environ.pop("GROUP_ARTIFACTS_DIR", None)
    else:
        os.environ["GROUP_ARTIFACTS_DIR"] = old


def test_screen_cell_reports_defined_for_s4(s4_artifacts_dir, monkeypatch):
    monkeypatch.setenv("GROUP_ARTIFACTS_DIR", str(s4_artifacts_dir))
    script = _load_script()
    result = script.screen_cell(S4_ORDER, S4_INDEX)
    assert result["status"] == "screened"
    assert result["verdict"] == "DEFINED"
    assert result["screen"]["support_fraction"] == pytest.approx(10 / 24)


def test_screen_cell_reports_artifact_missing_for_an_unexported_group(tmp_path, monkeypatch):
    monkeypatch.setenv("GROUP_ARTIFACTS_DIR", str(tmp_path))
    script = _load_script()
    result = script.screen_cell(999, 1)
    assert result["status"] == "artifact_missing"


def test_template_divergence_run_produces_a_measured_record(s4_run, s4_artifacts_dir, monkeypatch):
    monkeypatch.setenv("GROUP_ARTIFACTS_DIR", str(s4_artifacts_dir))
    script = _load_script()
    record = script.template_divergence_run(s4_run, threshold=0.0)
    assert record["status"] == "measured"
    assert record["group"] == {"order": S4_ORDER, "index": S4_INDEX, "name": "SmallGroup(24,12)"}
    assert record["screen"]["verdict"] == "DEFINED"

    for argument in ("left", "right"):
        divergence = record["occupancy"][argument]
        assert divergence["closest_template"] in {"null", "gcr", "coset"}
        for key in ("tv_to_null", "tv_to_gcr", "tv_to_coset"):
            assert isinstance(divergence[key], float)

    provenance = record["provenance"]
    assert provenance["config_hash"]
    assert provenance["checkpoint_sha256"]
    assert provenance["instrument_code_sha256"]
    assert provenance["group_artifact_path"].endswith("smallgroup_24_12.npz")
    json.dumps(record)  # the whole record must be JSON-serialisable


def test_template_divergence_run_skips_a_run_with_no_stable_checkpoint(
    s4_run, s4_artifacts_dir, monkeypatch
):
    monkeypatch.setenv("GROUP_ARTIFACTS_DIR", str(s4_artifacts_dir))
    script = _load_script()
    record = script.template_divergence_run(s4_run, threshold=0.99)
    assert record["status"] == "skipped"
    assert record["checkpoint_selection"]["reason"] is not None
    assert "occupancy" not in record


def test_template_divergence_run_reports_screen_undefined_without_loading_a_checkpoint(
    s4_run, monkeypatch
):
    """Point the driver at an artifact directory with no subgroup data for
    S4 (a bare re-export with ``subgroups=0``): the screen degrades to
    artifact-incomplete UNDEFINED, and the run is skipped before any
    checkpoint or model is touched."""
    bare = _build_s4_with_real_projectors()
    incomplete = GroupData(
        order=bare.order,
        index=bare.index,
        description=bare.description,
        element_labels=bare.element_labels,
        cayley_table=bare.cayley_table,
        conjugacy_classes=bare.conjugacy_classes,
        character_table=bare.character_table,
        frobenius_schur=bare.frobenius_schur,
        irreps=bare.irreps,
        isotypic_blocks=bare.isotypic_blocks,
        subgroups=(),
        left_cosets=(),
        provenance=bare.provenance,
    )
    directory = s4_run.parent.parent / "s4-incomplete-artifacts"
    _write_artifact(incomplete, directory / f"smallgroup_{S4_ORDER}_{S4_INDEX}.npz")
    monkeypatch.setenv("GROUP_ARTIFACTS_DIR", str(directory))
    script = _load_script()
    record = script.template_divergence_run(s4_run, threshold=0.0)
    assert record["status"] == "screen_undefined"
    assert record["screen"]["artifact_incomplete"] is True
    assert "occupancy" not in record


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def test_template_divergence_run_data_driven_populates_from_real_records(
    s4_run, s4_artifacts_dir, tmp_path, monkeypatch
):
    """Synthetic I-15 and GCR matrix-product records for S4's own cell
    (order 24, index 12, width 16 -- ``s4_run``'s config), placed at the
    exact filenames the driver looks for: block 2 ("two", rank 4) and block
    3 ("three", rank 9) cross the causal-use threshold; blocks 3 and 4
    ("three_p", rank 9) are product-carrying. The intersection is {3}, both
    full sets stay visible, and the comparison itself is built from the
    causally-used set {2, 3}."""
    monkeypatch.setenv("GROUP_ARTIFACTS_DIR", str(s4_artifacts_dir))
    coset_dir = tmp_path / "coset"
    gcr_matmul_dir = tmp_path / "gcr_matmul"
    _write_json(
        coset_dir / f"iso_ablation_{S4_ORDER}_{S4_INDEX}_w16.json",
        {
            "instrument": "isotypic-block-ablation-panel",
            "cell": {"order": S4_ORDER, "index": S4_INDEX, "width": 16},
            "blocks": [
                {
                    "block_index": 2,
                    "irrep_degree": 2,
                    "block_rank": 4,
                    "modes": {"zero": {"flip_fraction_over_random": {"mean": 0.2}}},
                },
                {
                    "block_index": 3,
                    "irrep_degree": 3,
                    "block_rank": 9,
                    "modes": {"zero": {"flip_fraction_over_random": {"mean": 0.3}}},
                },
                {
                    "block_index": 4,
                    "irrep_degree": 3,
                    "block_rank": 9,
                    "modes": {"zero": {"flip_fraction_over_random": {"mean": 0.01}}},
                },
            ],
        },
    )
    _write_json(
        gcr_matmul_dir / f"gcr_matmul_{S4_ORDER}_{S4_INDEX}_w16.json",
        {
            "instrument": "gcr_matmul",
            "group": {"order": S4_ORDER, "index": S4_INDEX, "name": "SmallGroup(24,12)"},
            "model": {"d_model": 16},
            "runs": [],
            "aggregate": {
                "irreps": [
                    {
                        "block_index": 3,
                        "irrep_degree": 3,
                        "mp_minus_bilinear_heldout": {"mean": 0.05},
                        "mp_favoured_fraction": 0.8,
                    },
                    {
                        "block_index": 4,
                        "irrep_degree": 3,
                        "mp_minus_bilinear_heldout": {"mean": 0.05},
                        "mp_favoured_fraction": 0.8,
                    },
                ]
            },
        },
    )

    script = _load_script()
    record = script.template_divergence_run(
        s4_run, threshold=0.0, coset_dir=coset_dir, gcr_matmul_dir=gcr_matmul_dir
    )
    assert record["status"] == "measured"

    for argument in ("left", "right"):
        data_driven = record["occupancy"][argument]["data_driven"]
        assert data_driven["status"] == "measured"
        assert data_driven["used_blocks_causal"] == [2, 3]
        assert data_driven["used_blocks_product_carrying"] == [3, 4]
        assert data_driven["used_blocks_intersection"] == [3]
        assert data_driven["iso_ablation_path"].endswith(
            f"iso_ablation_{S4_ORDER}_{S4_INDEX}_w16.json"
        )
        assert data_driven["gcr_matmul_path"].endswith(f"gcr_matmul_{S4_ORDER}_{S4_INDEX}_w16.json")

        comparison = data_driven["comparison"]
        assert comparison["used_blocks"] == [2, 3]
        assert isinstance(comparison["tv_to_used"], float)
        assert isinstance(comparison["tv_to_gcr_predicted"], float)
        assert isinstance(comparison["tv_to_coset_predicted"], float)
        assert comparison["closest_template"] in {"used", "gcr", "coset"}
        assert isinstance(comparison["separation"], float)

    json.dumps(record)  # the whole record, data-driven block included, must be JSON-serialisable


def test_template_divergence_run_data_driven_is_pending_when_records_are_absent(
    s4_run, s4_artifacts_dir, tmp_path, monkeypatch
):
    """Neither I-15 nor the GCR matrix-product fit has been run for this
    cell yet (an empty results tree): the data-driven block degrades to
    ``pending`` with both missing paths named, rather than failing the run
    or the template-only comparison it sits alongside."""
    monkeypatch.setenv("GROUP_ARTIFACTS_DIR", str(s4_artifacts_dir))
    coset_dir = tmp_path / "empty-coset"
    gcr_matmul_dir = tmp_path / "empty-gcr_matmul"
    script = _load_script()
    record = script.template_divergence_run(
        s4_run, threshold=0.0, coset_dir=coset_dir, gcr_matmul_dir=gcr_matmul_dir
    )
    assert record["status"] == "measured"
    for argument in ("left", "right"):
        data_driven = record["occupancy"][argument]["data_driven"]
        assert data_driven["status"] == "pending"
        assert len(data_driven["missing"]) == 2
        # the template-only comparison alongside it is unaffected
        assert isinstance(record["occupancy"][argument]["tv_to_null"], float)
    json.dumps(record)


def test_template_divergence_run_no_datadriven_flag_omits_the_block(
    s4_run, s4_artifacts_dir, monkeypatch
):
    monkeypatch.setenv("GROUP_ARTIFACTS_DIR", str(s4_artifacts_dir))
    script = _load_script()
    record = script.template_divergence_run(s4_run, threshold=0.0, datadriven=False)
    assert record["status"] == "measured"
    for argument in ("left", "right"):
        assert "data_driven" not in record["occupancy"][argument]


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

    _make_run("a_seed0", order=S4_ORDER, index=S4_INDEX, width=128, seed=0, completed=True)
    _make_run("b_seed1", order=S4_ORDER, index=S4_INDEX, width=128, seed=1, completed=True)
    _make_run("c_incomplete", order=S4_ORDER, index=S4_INDEX, width=128, seed=2, completed=False)

    cells = script.discover_cells(runs_dir)
    assert set(cells.keys()) == {(S4_ORDER, S4_INDEX, 128)}
    assert [p.name for p in cells[(S4_ORDER, S4_INDEX, 128)]] == ["a_seed0", "b_seed1"]


def test_cli_measure_writes_one_json_per_defined_cell(s4_run, s4_artifacts_dir, tmp_path):
    script = _load_script()
    out_dir = tmp_path / "template_divergence"
    runs_root = s4_run.parent
    code = script.main(
        [
            "measure",
            "--runs-dir",
            str(runs_root),
            "--out-dir",
            str(out_dir),
            "--artifacts-dir",
            str(s4_artifacts_dir),
            "--threshold",
            "0.0",
            "--max-seeds-per-cell",
            "50",
        ]
    )
    assert code == 0
    out_path = out_dir / f"template_divergence_{S4_ORDER}_{S4_INDEX}_w16.json"
    assert out_path.is_file()
    payload = json.loads(out_path.read_text())
    assert payload["cell"] == {"order": S4_ORDER, "index": S4_INDEX, "width": 16}
    assert payload["summary"]["n_measured"] == 1
    assert len(payload["runs"]) == 1


def test_cli_measure_exits_nonzero_when_a_defined_cell_run_is_skipped(
    s4_run, s4_artifacts_dir, tmp_path
):
    script = _load_script()
    out_dir = tmp_path / "template_divergence_skip"
    runs_root = s4_run.parent
    code = script.main(
        [
            "measure",
            "--runs-dir",
            str(runs_root),
            "--out-dir",
            str(out_dir),
            "--artifacts-dir",
            str(s4_artifacts_dir),
            "--threshold",
            "0.99",
        ]
    )
    assert code == 1
    payload = json.loads(
        (out_dir / f"template_divergence_{S4_ORDER}_{S4_INDEX}_w16.json").read_text()
    )
    assert payload["runs"][0]["status"] == "skipped"
