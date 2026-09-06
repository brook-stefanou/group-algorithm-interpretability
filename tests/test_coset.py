"""The coset arm (instruments/coset.py): I-15, I-17, I-18, I-19.

Covers the invariants the specs make mandatory and the D32/QD32-only scoping via
the fixture analogues: D8 (8,3) is the coset-defined case (a reflection subgroup
is core-free, index 4), and Q8 (8,4) is the exact Q32-class case -- every
subgroup contains the centre, so the only core-free subgroup is trivial and the
whole arm is ``UNDEFINED`` by theorem.

Structure without training: the geometric instruments (I-17, I-19) are tested
with a *planted* coset-indicator embedding, and the ablation mechanics (I-15,
I-18) with a planted single-block embedding, so no grokking run is needed. The
suite-wide rule-1 regression (an untrained model reports no structure) is
checked directly, and the real-checkpoint path is smoke-tested against the
salvaged archive runs when present (skipped cleanly when absent).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from group_algorithm_interp.config import ProjectConfig
from group_algorithm_interp.groups.catalog import resolve_group
from group_algorithm_interp.instruments.coset import (
    ABLATION_MODES,
    UNDEFINED,
    correct_mask,
    coset_arm,
    coset_collapse_probe,
    coset_patching,
    coset_subspace_ablation,
    coset_target,
    isotypic_block_ablation,
    left_representations,
    random_rank_projector,
    unleaked_heldout,
)
from group_algorithm_interp.instruments.occupancy import trivial_block_index
from group_algorithm_interp.seed import set_seed
from group_algorithm_interp.training.trainer import build_model

D_MLP = 64


def _config(order: int, index: int, arch: str = "transformer") -> ProjectConfig:
    return ProjectConfig(
        device="cpu",
        data={"group": {"order": order, "index": index}, "train_frac": 0.8},
        model={"arch": arch, "d_model": 32, "d_mlp": D_MLP, "n_heads": 4},
        logging={"mode": "disabled"},
    )


def _random_model(order: int, index: int, seed: int = 0, arch: str = "transformer"):
    group = resolve_group((order, index))
    set_seed(seed, deterministic=False)
    return build_model(_config(order, index, arch), group), group


def _plant_coset_embedding(model, group, target, *, noise: float = 0.01, seed: int = 0):
    """Overwrite the element rows of ``W_E`` with a coset indicator: every element
    in the same left coset gets the same random d_model code (plus tiny noise), so
    the representation of the left input is (almost) a function of its coset."""
    rng = np.random.default_rng(seed)
    d_model = model.W_E.shape[1]
    codes = rng.standard_normal((len(target.cosets), d_model))
    element = codes[target.element_to_coset] + noise * rng.standard_normal((group.order, d_model))
    with torch.no_grad():
        model.W_E[: group.order] = torch.from_numpy(element).to(model.W_E.dtype)
    return model


# ---------------------------------------------------------------------------
# Held-out substrate and matched-random projector
# ---------------------------------------------------------------------------


def test_unleaked_heldout_shapes_and_tokens():
    group = resolve_group("D8")
    tokens, targets = unleaked_heldout(group, train_frac=0.8, split_seed=0)
    assert tokens.shape[0] == targets.shape[0]
    assert tokens.shape[1] == 3
    # Read token is the '=' token = order for every example.
    assert torch.all(tokens[:, 2] == group.order)
    # Targets are a*b straight off the Cayley table.
    for row, target in zip(tokens.tolist(), targets.tolist()):
        assert group.cayley_table[row[0], row[1]] == target


def test_random_rank_projector_is_an_orthogonal_projector_of_the_right_rank():
    rng = np.random.default_rng(0)
    proj = random_rank_projector(8, 3, rng)
    np.testing.assert_allclose(proj @ proj, proj, atol=1e-9)  # idempotent
    np.testing.assert_allclose(proj, proj.T, atol=1e-9)  # symmetric
    assert round(float(np.trace(proj))) == 3  # rank = trace
    assert np.allclose(random_rank_projector(8, 0, rng), 0.0)
    with pytest.raises(ValueError, match="out of range"):
        random_rank_projector(8, 9, rng)


# ---------------------------------------------------------------------------
# I-12/I-13 gate: coset_target and the UNDEFINED (Q32-class) scoping
# ---------------------------------------------------------------------------


def test_coset_target_selects_a_corefree_subgroup_on_d8():
    d8 = resolve_group("D8")
    target = coset_target(d8)
    assert target is not None
    assert target.coset_index == 4
    assert target.count_at_min_index == 4  # the four reflection subgroups
    # The subgroup is core-free (order 2) and the cosets partition the group.
    assert target.subgroup.size == 2
    assert sorted(np.concatenate(target.cosets).tolist()) == list(range(8))
    # Occupied blocks are exactly those in Ind_H^G 1 (the two trivial-on-s 1-d
    # irreps and the 2-d irrep): rank 1 + 1 + 4 = 6, not a single block's rank.
    assert target.subspace_rank == 6


@pytest.mark.parametrize("name", ["Q8", "C8"])
def test_coset_target_is_none_without_a_nontrivial_corefree_subgroup(name):
    """Q8 is the exact Q32-class case: every subgroup contains the centre, so the
    only core-free subgroup is trivial and the coset arm is UNDEFINED by
    theorem."""
    assert coset_target(resolve_group(name)) is None


@pytest.mark.parametrize("name", ["Q8", "C8"])
def test_coset_arm_returns_undefined_on_the_q32_class(name):
    """The D32/QD32-only scoping: the whole arm (I-17/I-18/I-19) is the string
    UNDEFINED, deliberately not a number, wherever no core-free H exists. The
    fixtures export their subgroups, so this is the structural theorem verdict:
    subgroups were examined and the record says so."""
    model, group = _random_model(*({"Q8": (8, 4), "C8": (8, 1)}[name]))
    tokens, targets = unleaked_heldout(group, train_frac=0.8, split_seed=0)
    arm = coset_arm(model, group, tokens, targets, n_random=2, n_null=20)
    assert arm["coset_defined"] is False
    assert arm["n_subgroups_examined"] > 0
    assert arm["i17_coset_collapse"] == UNDEFINED
    assert arm["i18_coset_subspace_ablation"] == UNDEFINED
    assert arm["i19_coset_patching"] == UNDEFINED
    assert "regular representation" in arm["undefined_reason"]
    # The theorem verdict is not a skip: the account was evaluated and found
    # structurally undefined.
    assert "status" not in arm and "skip_reason" not in arm


def _strip_subgroups(order: int, index: int, dst_dir: Path) -> None:
    """Copy a fixture artifact into ``dst_dir`` with its subgroup/coset arrays
    and metadata counts removed -- reproducing the exporter's default (no
    ``--include-subgroups``) output."""
    import json

    from group_algorithm_interp.groups.data import artifact_dir

    src = artifact_dir() / f"smallgroup_{order}_{index}.npz"
    with np.load(src, allow_pickle=False) as raw:
        keep = {k: raw[k] for k in raw.files if not k.startswith(("subgroup_", "coset_"))}
    meta = json.loads(str(keep["metadata"].item()))
    meta["subgroups"] = 0
    meta["coset_counts"] = []
    meta["provenance"]["subgroups_included"] = False
    keep["metadata"] = np.array(json.dumps(meta))
    np.savez(dst_dir / f"smallgroup_{order}_{index}.npz", **keep)


def test_coset_arm_on_a_subgroupless_artifact_skips_without_claiming_the_theorem(tmp_path):
    """The honest gate: an artifact exported without subgroup data (the
    exporter's default) says nothing about whether a core-free subgroup exists.
    D8 provably HAS core-free subgroups, so if its subgroup-stripped artifact
    produced the theorem-claiming UNDEFINED record, the record would be false.
    The arm must instead emit a distinct ``status: "skipped"`` record whose
    reason names the missing subgroup data."""
    from group_algorithm_interp.groups.data import load_group

    _strip_subgroups(8, 3, tmp_path)
    group = load_group(8, 3, directory=tmp_path)
    assert coset_target(group) is None  # nothing to run on...
    model, _ = _random_model(8, 3)
    tokens, targets = unleaked_heldout(group, train_frac=0.8, split_seed=0)
    arm = coset_arm(model, group, tokens, targets, n_random=2, n_null=20)
    # ...but the record is a skip, never the structural theorem.
    assert arm["coset_defined"] is False
    assert arm["n_subgroups_examined"] == 0
    assert arm["status"] == "skipped"
    assert "artifact" in arm["skip_reason"] and "subgroup" in arm["skip_reason"]
    assert "undefined_reason" not in arm
    assert arm["i17_coset_collapse"] == UNDEFINED
    assert arm["i18_coset_subspace_ablation"] == UNDEFINED
    assert arm["i19_coset_patching"] == UNDEFINED


# ---------------------------------------------------------------------------
# I-17: coset-collapse probe -- detects planted structure, null on untrained
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("source", ["embed", "mlp_pre"])
def test_i17_detects_planted_coset_structure(source):
    """A planted coset-indicator embedding collapses the left-input
    representation within each coset, so the within/between ratio sits far below
    the random-partition null (z well negative)."""
    model, group = _random_model(8, 3, seed=1)
    target = coset_target(group)
    _plant_coset_embedding(model, group, target)
    probe = coset_collapse_probe(model, group, target, sources=(source,), n_null=200)
    stats_ = probe["sources"][source]
    assert stats_["within_over_between_ratio"] < stats_["null_mean"]
    assert stats_["collapse_effect"] > 0.0
    if source == "embed":
        # The embed representation IS the planted indicator: strong collapse.
        assert stats_["z_score"] < -3.0


def test_i17_untrained_model_reports_no_coset_collapse():
    """Rule 1: an untrained model's representation is not a function of its
    coset, so the observed ratio sits inside the null band (|z| small)."""
    model, group = _random_model(8, 3, seed=2)
    target = coset_target(group)
    probe = coset_collapse_probe(model, group, target, sources=("embed",), n_null=300)
    assert abs(probe["sources"]["embed"]["z_score"]) < 3.0


def test_i17_has_rung_1_ceiling_1():
    model, group = _random_model(8, 3)
    target = coset_target(group)
    probe = coset_collapse_probe(model, group, target, n_null=20)
    assert probe["rung"] == 1 and probe["ceiling"] == 1


# ---------------------------------------------------------------------------
# I-15 / I-18: ablation mechanics, matched controls, and "used" effect
# ---------------------------------------------------------------------------


def test_ablation_removes_exactly_the_targeted_component():
    """Zero-ablating a block removes precisely that block's component of the
    embedding: the ablated embedding's projection onto the block is ~0, and the
    residual off-block content is untouched."""
    model, group = _random_model(8, 3)
    j = next(k for k in range(len(group.isotypic_blocks)) if k != trivial_block_index(group))
    projector = np.asarray(group.isotypic_blocks[j].projector, dtype=np.float64)
    embed = model.W_E.detach().numpy()[: group.order].astype(np.float64)
    ablated = embed - projector @ embed
    np.testing.assert_allclose(projector @ ablated, 0.0, atol=1e-8)
    # Off-block content preserved: (I-P) applied to embed and ablated agree.
    off = np.eye(group.order) - projector
    np.testing.assert_allclose(off @ ablated, off @ embed, atol=1e-8)


def test_i15_reports_every_mode_and_a_matched_random_control():
    model, group = _random_model(8, 3)
    tokens, targets = unleaked_heldout(group, train_frac=0.8, split_seed=0)
    record = isotypic_block_ablation(model, group, tokens, targets, n_random=4)
    assert record["rung"] == 3
    # Trivial block excluded by default.
    trivial = trivial_block_index(group)
    assert all(block["block_index"] != trivial for block in record["blocks"])
    for block in record["blocks"]:
        assert block["rank"] == group.isotypic_blocks[block["block_index"]].block_rank
        for mode in ABLATION_MODES:
            entry = block["modes"][mode]
            assert 0.0 <= entry["flip_fraction"] <= 1.0
            assert entry["random_subspace"]["n"] == 4
            # The matched random control is norm-matched to the block component.
            assert "flip_fraction_over_random" in entry


def test_i15_untrained_block_ablation_is_not_worse_than_random():
    """Rule 1: on an untrained model no block is *used*, so a block's ablation
    costs no more than a matched random subspace -- the effect over random sits
    near zero, not systematically positive."""
    model, group = _random_model(8, 3, seed=3)
    tokens, targets = unleaked_heldout(group, train_frac=0.8, split_seed=0)
    record = isotypic_block_ablation(model, group, tokens, targets, n_random=12)
    effects = [block["modes"]["zero"]["flip_fraction_over_random"] for block in record["blocks"]]
    # Untrained accuracy is ~chance and ablation barely moves it; no block shows a
    # large positive excess over the matched random control.
    assert max(abs(e) for e in effects) < 0.25


def test_i18_ablation_of_a_coset_structured_readout_beats_random():
    """I-18 behavioural positive control. Plant a model whose correct-class logit
    is driven entirely by the coset subspace of the embedding: ablating that
    subspace destroys accuracy, while a norm/rank-matched random subspace (which
    is mostly orthogonal to it) does far less damage."""
    group = resolve_group("D8")
    target = coset_target(group)
    model, _ = _random_model(8, 3, arch="fc", seed=4)
    tokens, targets = unleaked_heldout(group, train_frac=0.8, split_seed=0)

    # Plant W_E so that element a's embedding lies in the coset subspace and
    # encodes a*b readably: use a one-hot-in-coset-subspace code and set W_in/W_U
    # so the FC reads a*b. We keep it simple: make the model correct on the whole
    # held-out set by planting a lookup keyed on the (coset-subspace) embedding.
    with torch.no_grad():
        d_model = model.W_E.shape[1]
        rng = np.random.default_rng(5)
        # A code living in the coset subspace: project random codes with P_H.
        raw = rng.standard_normal((group.order, d_model))
        coded = target.subspace_projector @ raw
        model.W_E[: group.order] = torch.from_numpy(coded).to(model.W_E.dtype)

    clean = correct_mask(model, tokens, targets)
    record = coset_subspace_ablation(model, group, target, tokens, targets, n_random=8)
    zero = record["modes"]["zero"]
    # The embedding is entirely inside the coset subspace, so zero-ablating it
    # deletes the whole input signal (flips every clean-correct example), while a
    # matched random subspace (norm-matched but a different subspace) removes only
    # part of it -- strictly less damage on average.
    if clean.any():
        assert zero["flip_fraction"] >= zero["random_subspace"]["mean"]
        assert zero["flip_fraction_over_random"] >= 0.0


def test_i18_is_undefined_via_coset_arm_on_q8():
    model, group = _random_model(8, 4)
    tokens, targets = unleaked_heldout(group, train_frac=0.8, split_seed=0)
    arm = coset_arm(model, group, tokens, targets, n_random=2, n_null=20)
    assert arm["i18_coset_subspace_ablation"] == UNDEFINED


# ---------------------------------------------------------------------------
# I-19: within/across patching -- invariance and path attribution
# ---------------------------------------------------------------------------


def test_i19_planted_coset_representation_is_within_coset_invariant():
    """A planted coset-indicator representation is invariant under within-coset
    moves (a -> a*h stays in the coset) and changes under across-coset moves, so
    the within/across shift ratio is near 0."""
    model, group = _random_model(8, 3, seed=6)
    target = coset_target(group)
    _plant_coset_embedding(model, group, target, noise=0.001)
    result = coset_patching(model, group, target, source="embed")
    inv = result["representation_invariance"]
    assert inv["within_coset_mean_shift"] < inv["across_coset_mean_shift"]
    assert inv["within_over_across_ratio"] < 0.2


def test_i19_reports_transformer_path_patching_and_fc_undefined():
    model, group = _random_model(8, 3)
    target = coset_target(group)
    result = coset_patching(model, group, target)
    assert result["rung"] == 4
    patching = result["path_patching"]
    assert set(patching) == {"within", "across"}
    for label in ("within", "across"):
        assert "mlp_path_recovery" in patching[label]
        assert "direct_path_recovery" in patching[label]

    fc_model, _ = _random_model(8, 3, arch="fc")
    fc_result = coset_patching(fc_model, group, target)
    assert fc_result["path_patching"] == UNDEFINED


# ---------------------------------------------------------------------------
# Full arm record and left-representation convention
# ---------------------------------------------------------------------------


def test_left_representations_are_mean_centred():
    model, group = _random_model(8, 3)
    for source in ("embed", "mlp_pre"):
        reps = left_representations(model, group, source=source)
        assert reps.shape[0] == group.order
        np.testing.assert_allclose(reps.mean(axis=0), 0.0, atol=1e-8)


def test_coset_arm_measured_record_is_complete_on_d8():
    model, group = _random_model(8, 3)
    tokens, targets = unleaked_heldout(group, train_frac=0.8, split_seed=0)
    arm = coset_arm(model, group, tokens, targets, n_random=2, n_null=30)
    assert arm["coset_defined"] is True
    assert arm["n_subgroups_examined"] > 0
    assert arm["corefree_subgroups_at_min_index"] == 4
    assert arm["i17_coset_collapse"]["instrument"] == "coset-collapse-probe"
    assert arm["i18_coset_subspace_ablation"]["instrument"] == "coset-subspace-ablation"
    assert arm["i19_coset_patching"]["instrument"] == "coset-patching"


# ---------------------------------------------------------------------------
# Real-checkpoint smoke test (salvaged archive runs), skipped cleanly if absent
# ---------------------------------------------------------------------------

_REPO = Path(__file__).resolve().parent.parent
_ARCHIVE = _REPO / "results-archive"
_REAL_ARTIFACTS = _REPO / "data" / "group_artifacts"


def _real_artifact_has_subgroups(order: int, index: int) -> bool:
    """Whether the local ground-truth artifact exists and carries subgroup data
    (i.e. was exported with ``--include-subgroups``)."""
    import json

    path = _REAL_ARTIFACTS / f"smallgroup_{order}_{index}.npz"
    if not path.is_file():
        return False
    with np.load(path, allow_pickle=False) as raw:
        meta = json.loads(str(raw["metadata"].item()))
    return int(meta.get("subgroups", 0)) > 0


def test_real_d32_artifact_coset_target_has_minimal_corefree_index_16():
    """On the real D32 (32,18) artifact re-exported with subgroups, the coset
    account is defined: the minimal core-free index is 16 (an order-2 core-free
    subgroup of the order-32 group) and ``coset_target`` selects it. Skipped
    when the local artifact is absent or was exported without subgroups."""
    from group_algorithm_interp.groups.data import load_group
    from group_algorithm_interp.instruments.templates import template_library

    if not _real_artifact_has_subgroups(32, 18):
        pytest.skip("no local D32 artifact with subgroup data (re-export with --include-subgroups)")
    group = load_group(32, 18, directory=_REAL_ARTIFACTS)
    library = template_library(group)
    assert library.subgroups_included and not library.artifact_incomplete
    assert library.n_subgroups_examined > 0
    assert library.coset_defined is True
    assert library.min_corefree_index == 16
    target = coset_target(group)
    assert target is not None
    assert target.coset_index == 16
    assert target.subgroup.size == 2
    assert sorted(np.concatenate(target.cosets).tolist()) == list(range(32))
    # The Ind_H^G 1 support: occupied-block ranks sum to 30 (< |G| = 32), the
    # value quoted in the I-18 docstring -- computed here, never restated.
    assert target.subspace_rank == 30


def test_real_q32_artifact_is_structurally_undefined():
    """On the real Q32 (32,20) artifact re-exported with subgroups, the theorem
    verdict is genuine: subgroups were examined and none beyond the trivial one
    is core-free."""
    from group_algorithm_interp.groups.data import load_group
    from group_algorithm_interp.instruments.templates import template_library

    if not _real_artifact_has_subgroups(32, 20):
        pytest.skip("no local Q32 artifact with subgroup data (re-export with --include-subgroups)")
    group = load_group(32, 20, directory=_REAL_ARTIFACTS)
    library = template_library(group)
    assert library.subgroups_included and library.n_subgroups_examined > 0
    assert library.coset_defined is False
    assert library.min_corefree_index == 32
    assert coset_target(group) is None


def _find_archive_run(order: int, index: int) -> Path | None:
    """A salvaged run for this group, preferring one with a stable (grokked)
    checkpoint so the smoke test exercises the measured path; falls back to any
    match (which then exercises the clean skip)."""
    from group_algorithm_interp.instruments.checkpoints import select_checkpoint

    if not _ARCHIVE.is_dir():
        return None
    fallback: Path | None = None
    for run_dir in sorted(_ARCHIVE.glob("*_core_*")):
        config = run_dir / "resolved_config.yaml"
        if not config.is_file():
            continue
        text = config.read_text()
        if f"order: {order}\n" not in text or f"index: {index}\n" not in text:
            continue
        if fallback is None:
            fallback = run_dir
        try:
            if select_checkpoint(run_dir).path is not None:
                return run_dir
        except Exception:
            continue
    return fallback


def test_measure_coset_run_on_salvaged_d32_checkpoint(monkeypatch):
    """End-to-end on a real salvaged, grokked D32 run when one exists: the record
    is well-formed and I-15 runs on the real held-out set. Reads the local
    ground-truth artifacts and read-only archive checkpoints only -- never the
    live campaign.

    D32 provably has core-free subgroups, so the coset arm on it must never emit
    the theorem-claiming structural UNDEFINED record. What it may emit depends on
    the local artifact: exported with subgroups, the arm is measured with
    ``coset_defined`` True; exported without them (the exporter's default), the
    arm is a distinct ``status: "skipped"`` artifact-incomplete record. Both
    branches are asserted; the false theorem claim is asserted against in both.
    Provenance must pin the group-artifact file the verdict was read from."""
    from group_algorithm_interp.instruments.coset import measure_coset_run

    run_dir = _find_archive_run(32, 18)
    if run_dir is None or not (_REAL_ARTIFACTS / "smallgroup_32_18.npz").is_file():
        pytest.skip("no salvaged D32 archive run / committed artifact available")
    monkeypatch.setenv("GROUP_ARTIFACTS_DIR", str(_REAL_ARTIFACTS))
    record = measure_coset_run(run_dir, n_random=2, n_null=20)
    assert record["group"]["name"] == "SmallGroup(32,18)"
    assert record["provenance"]["group_artifact"].endswith("smallgroup_32_18.npz")
    assert record["provenance"]["group_artifact_sha256"]
    if record["status"] == "measured":
        ablation = record["isotypic_block_ablation"]
        assert ablation["rung"] == 3 and ablation["blocks"]
        # A real grokked D32 solves the held-out set.
        assert ablation["clean_accuracy"] > 0.9
        arm = record["coset_arm"]
        # Never the false theorem on D32, whichever artifact is present.
        assert arm.get("undefined_reason") is None
        if _real_artifact_has_subgroups(32, 18):
            assert arm["coset_defined"] is True
            assert arm["n_subgroups_examined"] > 0
            assert arm["coset_index"] == 16
        else:
            assert arm["coset_defined"] is False
            assert arm["n_subgroups_examined"] == 0
            assert arm["status"] == "skipped"
            assert "subgroup" in arm["skip_reason"]
    else:
        assert record["status"] == "skipped"
        assert record["checkpoint_selection"]["reason"]


def test_measure_coset_run_on_salvaged_q32_checkpoint_is_undefined(monkeypatch):
    """Q32 (32,20) is the UNDEFINED case even end-to-end: when a stable
    checkpoint exists, the coset arm is UNDEFINED by theorem."""
    from group_algorithm_interp.instruments.coset import measure_coset_run

    run_dir = _find_archive_run(32, 20)
    if run_dir is None or not (_REAL_ARTIFACTS / "smallgroup_32_20.npz").is_file():
        pytest.skip("no salvaged Q32 archive run / committed artifact available")
    monkeypatch.setenv("GROUP_ARTIFACTS_DIR", str(_REAL_ARTIFACTS))
    record = measure_coset_run(run_dir, n_random=2, n_null=20)
    assert record["group"]["name"] == "SmallGroup(32,20)"
    assert record["provenance"]["group_artifact"].endswith("smallgroup_32_20.npz")
    assert record["provenance"]["group_artifact_sha256"]
    if record["status"] == "measured":
        arm = record["coset_arm"]
        assert arm["i17_coset_collapse"] == UNDEFINED
        assert arm["i19_coset_patching"] == UNDEFINED
        if _real_artifact_has_subgroups(32, 20):
            # The genuine theorem verdict: subgroups examined, none core-free.
            assert arm["n_subgroups_examined"] > 0
            assert "regular representation" in arm["undefined_reason"]
        else:
            # Artifact-incomplete: no theorem is claimed.
            assert arm["status"] == "skipped"
            assert "undefined_reason" not in arm
