"""Tests for ``scripts/derive_within_pair_contrasts.py`` -- the within-pair
contrast derivation behind the portfolio draft's matched-pair numbers.

Two layers, both offline (no network, no GAP, no W&B): the derivation reads
only the already-committed ``results/*.json`` per-cell files.

* Pure-function unit tests on synthetic fixtures: the permutation test, the
  Benjamini-Hochberg procedure, the binomial tail, and the per-instrument
  extractors.
* End-to-end tests against the real committed corpus under ``results/``,
  pinning the specific numbers the draft cites (the recruited-dimension
  FS-flip control value, the three used-set-sufficiency Benjamini-Hochberg
  survivors, and the clean nulls on the c1_matched pairs).

The script is an executable entry point, not part of the installed package,
so it is loaded by file path (mirrors ``tests/test_derive_falsifiers.py``).
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

_REPO = Path(__file__).resolve().parent.parent
_SCRIPT_PATH = _REPO / "scripts" / "derive_within_pair_contrasts.py"
_COMMITTED_OUTPUT = _REPO / "results" / "within_pair_contrasts" / "within_pair_contrasts.json"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("derive_within_pair_contrasts", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


dwpc = _load_module()


# --------------------------------------------------------------------------- #
# Pure-function unit tests                                                    #
# --------------------------------------------------------------------------- #
def test_sign_flip_permutation_test_all_same_sign_is_significant() -> None:
    # every diff strongly positive -> only the all-positive sign pattern (and
    # its mirror) is at least as extreme -> small exact p-value.
    diffs = [1.0, 1.1, 0.9, 1.2, 1.05]
    result = dwpc.sign_flip_permutation_test(diffs, exact_max_n=16)
    assert result["exact"] is True
    assert result["n_permutations"] == 32
    assert result["p_two_sided"] < 0.10


def test_sign_flip_permutation_test_symmetric_diffs_not_significant() -> None:
    # diffs symmetric around zero -> many sign patterns are at least as extreme.
    diffs = [1.0, -1.0, 1.0, -1.0, 0.1, -0.1]
    result = dwpc.sign_flip_permutation_test(diffs, exact_max_n=16)
    assert result["exact"] is True
    assert result["p_two_sided"] > 0.5


def test_sign_flip_permutation_test_monte_carlo_path_matches_exact_roughly() -> None:
    diffs = [
        1.0,
        1.1,
        0.9,
        1.2,
        1.05,
        0.95,
        1.15,
        1.0,
        0.85,
        1.3,
        0.8,
        1.25,
        1.0,
        0.9,
        1.1,
        1.0,
        1.05,
    ]
    exact_forced_small_n = dwpc.sign_flip_permutation_test(diffs[:10], exact_max_n=16)
    monte_carlo = dwpc.sign_flip_permutation_test(
        diffs[:10], exact_max_n=4, n_resamples=20_000, seed=0
    )
    assert exact_forced_small_n["exact"] is True
    assert monte_carlo["exact"] is False
    # both should land in the same rough significance ballpark for this
    # strongly-one-sided fixture.
    assert abs(exact_forced_small_n["p_two_sided"] - monte_carlo["p_two_sided"]) < 0.05


def test_benjamini_hochberg_classic_example() -> None:
    # Standard textbook check (BH step-up, q=0.05): thresholds are
    # i/5 * 0.05 = 0.01, 0.02, 0.03, 0.04, 0.05 for the sorted p-values.
    # 0.005 <= 0.01, 0.011 <= 0.02, 0.030 <= 0.03 all hold; 0.09 and 0.25
    # exceed their thresholds, so the largest surviving rank is 3.
    pvals = [0.005, 0.011, 0.03, 0.09, 0.25]
    reject = dwpc.benjamini_hochberg(pvals, 0.05)
    assert reject == [True, True, True, False, False]


def test_benjamini_hochberg_all_null_rejects_nothing() -> None:
    pvals = [0.9, 0.5, 0.7, 0.3, 0.6]
    reject = dwpc.benjamini_hochberg(pvals, 0.05)
    assert not any(reject)


def test_binomial_upper_tail_p_edges() -> None:
    assert dwpc.binomial_upper_tail_p(10, 0, 0.05) == pytest.approx(1.0)
    assert abs(dwpc.binomial_upper_tail_p(10, 10, 0.05) - 0.05**10) < 1e-15
    # a middling case: P(X>=k) should be less likely as k grows
    p5 = dwpc.binomial_upper_tail_p(69, 5, 0.05)
    p22 = dwpc.binomial_upper_tail_p(69, 22, 0.05)
    assert p22 < p5


def test_noise_floor_summary_perfect_correlation_zero_paired_sd() -> None:
    a = [1.0, 2.0, 3.0, 4.0]
    b = [x - 0.5 for x in a]  # constant offset -> perfectly correlated, zero-variance diff
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    out = dwpc.noise_floor_summary(a, b, diffs)
    assert out["sd_paired_diff"] == 0.0
    assert abs(out["mean_diff"] - 0.5) < 1e-12
    assert out["ab_correlation"] > 0.999


def test_extract_used_set_audit_necessity_skips_zero_used_blocks() -> None:
    runs = [
        {
            "status": "measured",
            "seed": 0,
            "used_set_audit": {"n_necessary_blocks": 0, "n_used_blocks": 0},
        },
        {
            "status": "measured",
            "seed": 1,
            "used_set_audit": {"n_necessary_blocks": 2, "n_used_blocks": 4},
        },
        {
            "status": "skipped",
            "seed": 2,
            "used_set_audit": {"n_necessary_blocks": 1, "n_used_blocks": 1},
        },
    ]
    out = dwpc.extract_used_set_audit_necessity(runs)
    assert out == {1: 0.5}


def test_extract_coset_quotient_uses_decisive_subgroup() -> None:
    runs = [
        {
            "status": "measured",
            "seed": 0,
            "coset_quotient_route": {
                "defined": True,
                "decisive_subgroup_index": 34,
                "quotients": [
                    {"subgroup_index": 32, "necessity": {"coset_accuracy_drop_over_random": 0.1}},
                    {"subgroup_index": 34, "necessity": {"coset_accuracy_drop_over_random": 0.7}},
                ],
            },
        },
        {
            "status": "measured",
            "seed": 1,
            "coset_quotient_route": {"defined": False},
        },
    ]
    out = dwpc.extract_coset_quotient(runs)
    assert out == {0: 0.7}


def test_glob_cell_files_matches_order_index_width() -> None:
    files = dwpc.glob_cell_files(_REPO / "results" / "occupancy", (64, 74))
    assert "w128" in files
    assert files["w128"].name == "occupancy_64_74_w128.json"
    # a different index at the same order must not match
    other = dwpc.glob_cell_files(_REPO / "results" / "occupancy", (64, 7))
    assert other == {}


# --------------------------------------------------------------------------- #
# End-to-end against the real committed corpus                                #
# --------------------------------------------------------------------------- #
def test_committed_output_exists_and_has_expected_top_level_schema() -> None:
    assert _COMMITTED_OUTPUT.exists(), (
        "committed within-pair-contrasts output must ship in results/"
    )
    data = json.loads(_COMMITTED_OUTPUT.read_text())
    assert data["instrument"] == "within-pair-contrasts"
    for key in (
        "provenance",
        "headline_fields",
        "pair_categories",
        "contrasts",
        "multiple_looks",
        "used_set_audit_sufficiency_bh_survivors_q0.05",
        "used_set_audit_sufficiency_bh_survivors_c1_matched_only",
        "used_set_stress_test",
    ):
        assert key in data
    assert data["provenance"]["input_files_sha256"], "provenance must record input file hashes"


def test_committed_output_recruited_dimension_fs_flip_control_value() -> None:
    data = json.loads(_COMMITTED_OUTPUT.read_text())
    match = next(
        c
        for c in data["contrasts"]
        if c["instrument"] == "recruited_dimension" and c["pair"].startswith("(104,4)/(104,6)")
    )
    assert match["category"] == "fs_flip_control"
    diff = match["diff_a_minus_b"]
    assert diff["mean"] == pytest.approx(-1.534, abs=0.01)
    lo, hi = diff["bootstrap_ci_95"]
    assert lo == pytest.approx(-1.954, abs=0.01)
    assert hi == pytest.approx(-1.135, abs=0.01)
    assert diff["clears_zero"] is True


def test_committed_output_used_set_bh_survivors_are_exactly_the_three_c1_matched_pairs() -> None:
    data = json.loads(_COMMITTED_OUTPUT.read_text())
    survivors = data["used_set_audit_sufficiency_bh_survivors_c1_matched_only"]
    pairs = sorted(c["pair"] for c in survivors)
    assert pairs == sorted(
        [
            "(27,3)/(27,4) w512",
            "(64,74)/(64,80) w128",
            "(64,228)/(64,229) w128",
        ]
    )
    for c in survivors:
        assert c["category"] == "c1_matched"
        # each margin is on the order of a couple of accuracy points.
        assert 0.01 < abs(c["diff_a_minus_b"]["mean"]) < 0.05


def test_committed_output_occupancy_and_recruited_dimension_c1_matched_all_span_zero() -> None:
    data = json.loads(_COMMITTED_OUTPUT.read_text())
    for c in data["contrasts"]:
        if (
            c["status"] == "computed"
            and c["category"] == "c1_matched"
            and c["instrument"] in ("occupancy", "recruited_dimension")
        ):
            assert c["diff_a_minus_b"]["clears_zero"] is False, (
                f"{c['instrument']} {c['pair']} unexpectedly clears zero on a genuine-clean C1 pair"
            )


def test_recompute_matches_committed_output_for_a_sample_entry() -> None:
    """Re-derive one entry from scratch (not read from the committed file) and
    check it agrees with what's committed -- the byte-reproducibility claim,
    spot-checked rather than diffing the whole file (which also carries a
    live git-dirty flag)."""
    contrasts, _ = dwpc.compute_contrasts(_REPO / "results", bootstrap_seed=0, permutation_seed=0)
    recomputed = next(
        c
        for c in contrasts
        if c["instrument"] == "used_set_audit_sufficiency" and c["pair"] == "(64,228)/(64,229) w128"
    )
    committed = json.loads(_COMMITTED_OUTPUT.read_text())
    committed_entry = next(
        c
        for c in committed["contrasts"]
        if c["instrument"] == "used_set_audit_sufficiency" and c["pair"] == "(64,228)/(64,229) w128"
    )
    assert recomputed["diff_a_minus_b"]["mean"] == committed_entry["diff_a_minus_b"]["mean"]
    assert (
        recomputed["diff_a_minus_b"]["bootstrap_ci_95"]
        == committed_entry["diff_a_minus_b"]["bootstrap_ci_95"]
    )
    assert (
        recomputed["permutation_test"]["p_two_sided"]
        == committed_entry["permutation_test"]["p_two_sided"]
    )


def test_cli_invocation_reproduces_key_numbers(tmp_path: Path) -> None:
    output = tmp_path / "out.json"
    subprocess.run(
        [sys.executable, str(_SCRIPT_PATH), "--output", str(output)],
        check=True,
        cwd=_REPO,
    )
    data = json.loads(output.read_text())
    survivors = sorted(
        c["pair"] for c in data["used_set_audit_sufficiency_bh_survivors_c1_matched_only"]
    )
    assert survivors == sorted(
        ["(27,3)/(27,4) w512", "(64,74)/(64,80) w128", "(64,228)/(64,229) w128"]
    )
