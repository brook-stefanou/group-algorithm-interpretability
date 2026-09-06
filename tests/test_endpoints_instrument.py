"""Tier-0 endpoint layer (instruments/endpoints.py): I-01/I-02/I-04/I-06.

Covers the estimation-first endpoints the study reports: epochs-to-grok with
the pre-registered censoring rule (I-01), the realised-leak covariate read from
the manifest (I-02), the paired within-pair difference estimators with bootstrap
CIs and no significance gate (I-04), and the per-run measurement vector with
provenance and no verdict label (I-06). Each estimator is checked against a
hand computation. Two smoke tests run the parser and endpoints on real salvaged
v1 runs (skipped when ``results-archive/`` is absent, e.g. in CI).
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

from group_algorithm_interp.config import ProjectConfig
from group_algorithm_interp.instruments.endpoints import (
    GrokTime,
    chance_accuracy,
    epochs_to_grok,
    measurement_vector,
    metric_series,
    paired_difference,
    paired_grok_difference,
    paired_sign_flip_permutation,
    realised_leak_covariate,
    within_pair_endpoints,
)

_ARCHIVE = Path(__file__).resolve().parent.parent / "results-archive"
_ARCHIVE_GROKKED = _ARCHIVE / "2026-07-15_125014_403192_core_aa3553"
_ARCHIVE_CENSORED = _ARCHIVE / "2026-07-15_125014_165786_core_7d2298"


# ---------------------------------------------------------------------------
# I-01: chance anchor and epochs-to-grok
# ---------------------------------------------------------------------------


def test_chance_accuracy_is_one_over_order():
    assert chance_accuracy(8) == 0.125
    with pytest.raises(ValueError):
        chance_accuracy(0)


def test_epochs_to_grok_onset_is_first_of_the_sustained_streak():
    # Below the bar, then a 5-long streak beginning at epoch 20.
    series = [(e, 0.5) for e in range(0, 20)] + [(20 + i, 0.995) for i in range(5)]
    grok = epochs_to_grok(series, ceiling=1000, threshold=0.99, sustain=5)
    assert grok.epoch == 20
    assert grok.censored is False
    assert grok.grok_value == 0.995


def test_epochs_to_grok_needs_the_full_sustain_and_takes_the_later_streak():
    # A 4-long streak (too short) at 10, drop, then a real 5-long streak at 30.
    series = (
        [(e, 0.2) for e in range(0, 10)]
        + [(10 + i, 0.995) for i in range(4)]
        + [(14, 0.2)]
        + [(30 + i, 0.995) for i in range(5)]
    )
    grok = epochs_to_grok(series, ceiling=1000, threshold=0.99, sustain=5)
    assert grok.epoch == 30


def test_epochs_to_grok_bar_is_strict_and_nan_never_groks():
    at_bar = [(i, 0.99) for i in range(10)]  # exactly 0.99 is not > 0.99
    assert epochs_to_grok(at_bar, ceiling=100, sustain=5).censored is True
    nans = [(i, float("nan")) for i in range(10)]
    assert epochs_to_grok(nans, ceiling=100, sustain=5).censored is True


def test_epochs_to_grok_censors_when_streak_never_reached():
    series = [(i, 0.995 if i % 2 == 0 else 0.5) for i in range(0, 100)]
    grok = epochs_to_grok(series, ceiling=100, threshold=0.99, sustain=5)
    assert grok.censored is True
    assert grok.epoch is None
    assert grok.reached_ceiling is True  # last recorded epoch 99 >= ceiling-1


# ---------------------------------------------------------------------------
# I-04: paired difference estimators (hand-computed)
# ---------------------------------------------------------------------------


def test_paired_sign_flip_permutation_hand_value():
    # diffs = [1, 1, 1]: observed |mean| = 1. Of the 8 sign assignments only the
    # all-plus and all-minus reach |mean| >= 1, so p = 2/8 = 0.25.
    result = paired_sign_flip_permutation([1.0, 1.0, 1.0])
    assert result["exact"] == 1.0
    assert result["n_permutations"] == 8.0
    assert result["p_two_sided"] == pytest.approx(0.25)


def test_paired_sign_flip_permutation_sampled_branch_is_deterministic_and_sane():
    # n = 50 > exact_max_n (22): the campaign's real pair size, and the branch
    # with no prior test coverage anywhere. A strong, all-positive signal (a
    # real effect) should land at the sampled floor p = 1/(n_resamples + 1),
    # and the result must be exactly reproducible for a fixed seed.
    diffs = [10.0 + 0.1 * i for i in range(50)]
    result_1 = paired_sign_flip_permutation(diffs, seed=0, n_resamples=2000)
    result_2 = paired_sign_flip_permutation(diffs, seed=0, n_resamples=2000)
    assert result_1 == result_2  # same seed -> bit-identical result
    assert result_1["exact"] == 0.0
    assert result_1["n_permutations"] == 2000.0
    assert result_1["p_two_sided"] == pytest.approx(1.0 / 2001.0)

    # A different seed still explores the same null and lands at the same
    # floor for this strongly-separated signal, but need not match exactly.
    result_3 = paired_sign_flip_permutation(diffs, seed=1, n_resamples=2000)
    assert result_3["p_two_sided"] == pytest.approx(1.0 / 2001.0)

    # A null (mean-zero, symmetric) signal should NOT sit at the floor.
    null_diffs = [((-1.0) ** i) * 1.0 for i in range(50)]
    result_null = paired_sign_flip_permutation(null_diffs, seed=0, n_resamples=2000)
    assert result_null["p_two_sided"] > 1.0 / 2001.0


def test_paired_difference_hand_values():
    result = paired_difference([2.0, 3.0, 4.0], [1.0, 1.0, 1.0])
    assert result["per_seed_difference"] == [1.0, 2.0, 3.0]
    assert result["mean_difference"] == pytest.approx(2.0)
    assert result["median_difference"] == pytest.approx(2.0)
    assert result["std_difference"] == pytest.approx(1.0)
    assert result["standardised_effect_dz"] == pytest.approx(2.0)
    # All three differences positive: sign test n=3, exact two-sided p = 2/8.
    assert result["sign_test"]["n"] == 3.0
    assert result["sign_test"]["p_two_sided"] == pytest.approx(0.25)
    lo, hi = result["bootstrap_ci_95"]
    assert 1.0 <= lo <= hi <= 3.0


def test_paired_difference_rejects_unequal_lengths():
    with pytest.raises(ValueError):
        paired_difference([1.0, 2.0], [1.0])


def test_paired_difference_rejects_nan_and_none():
    # A fallback member's unleaked accuracy is NaN; paired_difference must
    # raise a clear error rather than crash cryptically in the bootstrap or
    # silently emit a NaN mean/CI.
    with pytest.raises(ValueError, match="non-finite or missing"):
        paired_difference([1.0, float("nan")], [0.5, 0.5])
    with pytest.raises(ValueError, match="non-finite or missing"):
        paired_difference([1.0, 0.9], [0.5, None])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="non-finite or missing"):
        paired_difference([1.0, float("inf")], [0.5, 0.5])


def _grok(epoch, ceiling=1000, *, reached_ceiling=True, metric="val/unleaked_accuracy"):
    return GrokTime(
        epoch=epoch,
        censored=epoch is None,
        ceiling=ceiling,
        metric=metric,
        threshold=0.99,
        sustain=5,
        grok_value=None if epoch is None else 0.995,
        n_evaluated_rows=ceiling,
        max_epoch_recorded=ceiling - 1,
        reached_ceiling=reached_ceiling,
    )


def test_paired_grok_difference_applies_the_censoring_rule():
    a = [_grok(100), _grok(60), _grok(None), _grok(50), _grok(None)]
    b = [_grok(50), _grok(80), _grok(50), _grok(None), _grok(None)]
    result = paired_grok_difference(a, b)

    assert result["n_pairs"] == 5
    assert result["n_both_grokked"] == 2
    assert result["n_a_censored_only"] == 1  # pair 3: a censored, b grokked
    assert result["n_b_censored_only"] == 1  # pair 4: b censored, a grokked
    assert result["n_both_censored_ties"] == 1
    assert result["censoring_fraction_a"] == pytest.approx(0.4)
    assert result["censoring_fraction_b"] == pytest.approx(0.4)

    # Signed pairs for the sign test: [+50, -20, +1, -1] -> 2 pos, 2 neg.
    assert result["sign_test"]["n"] == 4.0
    assert result["sign_test"]["n_positive"] == 2.0

    numeric = result["numeric_both_grokked"]
    assert numeric["per_seed_difference"] == [50.0, -20.0]
    assert numeric["mean_difference"] == pytest.approx(15.0)
    assert numeric["n_excluded_censored"] == 3


def test_paired_grok_difference_all_both_censored_is_all_ties():
    a = [_grok(None), _grok(None)]
    b = [_grok(None), _grok(None)]
    result = paired_grok_difference(a, b)
    assert result["n_both_censored_ties"] == 2
    assert result["sign_test"]["n"] == 0.0
    assert result["numeric_both_grokked"]["bootstrap_ci_95"] is None


def test_paired_grok_difference_truncated_censored_member_is_excluded_not_signed():
    # Pair 0: a is censored but its log stopped short of the ceiling (a crash) --
    # its true grok time is unknown, so it must not be counted as a known
    # "slower" sign. Pair 1: both grokked normally, an ordinary numeric diff.
    a = [_grok(None, reached_ceiling=False), _grok(60)]
    b = [_grok(50), _grok(80)]
    result = paired_grok_difference(a, b)

    assert result["n_truncated_pairs"] == 1
    assert result["n_a_censored_only"] == 0
    assert result["n_b_censored_only"] == 0
    assert result["n_both_grokked"] == 1
    # Only the both-grokked pair contributes a sign; the truncated pair is gone.
    assert result["sign_test"]["n"] == 1.0
    numeric = result["numeric_both_grokked"]
    assert numeric["per_seed_difference"] == [-20.0]


def test_paired_grok_difference_truncated_b_member_is_excluded_not_signed():
    a = [_grok(50)]
    b = [_grok(None, reached_ceiling=False)]
    result = paired_grok_difference(a, b)
    assert result["n_truncated_pairs"] == 1
    assert result["n_b_censored_only"] == 0
    assert result["sign_test"]["n"] == 0.0
    assert result["n_pairs"] == 1


def test_paired_grok_difference_rejects_mixed_ceiling():
    a = [_grok(100), _grok(60)]
    b = [_grok(50, ceiling=500), _grok(80, ceiling=500)]
    with pytest.raises(ValueError, match="ceiling"):
        paired_grok_difference(a, b)


def test_paired_grok_difference_rejects_mixed_metric():
    a = [_grok(100, metric="val/unleaked_accuracy"), _grok(60, metric="val/accuracy")]
    b = [_grok(50), _grok(80)]
    with pytest.raises(ValueError, match="metric"):
        paired_grok_difference(a, b)


# ---------------------------------------------------------------------------
# I-02: realised-leak covariate
# ---------------------------------------------------------------------------


def test_realised_leak_covariate_identity_and_residual():
    manifest = {
        "dataset": {
            "leakage": {
                "transpose_leak_fraction": 0.30,
                "commuting_probability": 0.25,
                "test_size": 200,
                "unleaked_test_size": 140,
                "unleaked_empty": False,
                "generalize_metric": "unleaked_accuracy",
            }
        }
    }
    cov = realised_leak_covariate(manifest, train_frac=0.8)
    assert cov["leak_estimate_trainfrac_times_commuting"] == pytest.approx(0.2)
    assert cov["realised_minus_estimate"] == pytest.approx(0.1)
    assert cov["unleaked_test_size"] == 140

    # No leakage block -> covariate degrades to Nones, never raises.
    empty = realised_leak_covariate({}, train_frac=0.8)
    assert empty["transpose_leak_fraction"] is None
    assert empty["leak_estimate_trainfrac_times_commuting"] is None


# ---------------------------------------------------------------------------
# I-06: per-run measurement vector (synthesised run dirs)
# ---------------------------------------------------------------------------


def _write_run(
    run_dir: Path,
    *,
    order: int = 8,
    index: int = 3,
    seed: int = 0,
    epochs: int = 200,
    unleaked_curve,
    leak_fraction: float = 0.30,
    commuting: float = 0.25,
    unleaked_empty: bool = False,
    with_final_pt: bool = True,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    config = ProjectConfig(
        device="cpu",
        seed=seed,
        data={"group": {"order": order, "index": index}, "train_frac": 0.8},
        model={"d_model": 16, "d_mlp": 32, "n_heads": 1},
        optim={"epochs": epochs},
        # Final-gate rule (gate final.pt on its own last row), so the synthesised
        # final.pt is the selectable checkpoint without needing window snapshots.
        snapshot={"final_window_epochs": 0},
        logging={"mode": "disabled"},
    )
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config.model_dump()))
    manifest = {
        "run_id": run_dir.name,
        "provenance": {
            "git_commit": "deadbeef",
            "config_hash": "abc123",
            "config_group_hash": "grp999",
            "campaign_id": None,
        },
        "dataset": {
            "spec_hash": "data777",
            "leakage": {
                "transpose_leak_fraction": leak_fraction,
                "commuting_probability": commuting,
                "test_size": 200,
                "unleaked_test_size": 0 if unleaked_empty else 140,
                "unleaked_empty": unleaked_empty,
                "generalize_metric": "unleaked_accuracy",
            },
        },
    }
    (run_dir / "manifest.yaml").write_text(yaml.safe_dump(manifest))
    lines = []
    for epoch, unleaked in unleaked_curve:
        row = {
            "train/loss": 0.01,
            "train/accuracy": 1.0,
            "val/loss": 0.2,
            "val/accuracy": min(1.0, unleaked + 0.02),
            "val/unleaked_accuracy": unleaked,
            "val/weight_norm": 10.0,
        }
        lines.append(f"epoch {epoch} | {row}")
    (run_dir / "run.log").write_text("\n".join(lines) + "\n")
    if with_final_pt:
        (run_dir / "checkpoints").mkdir(exist_ok=True)
        (run_dir / "checkpoints" / "final.pt").write_bytes(b"stub")
    return run_dir


def _grok_curve(onset: int, total: int) -> list[tuple[int, float]]:
    return [(e, 0.995 if e >= onset else 0.4) for e in range(total)]


def test_measurement_vector_grokked_run(tmp_path):
    run = _write_run(tmp_path / "grokked", unleaked_curve=_grok_curve(onset=30, total=200))
    record = measurement_vector(run)

    assert record["status"] == "measured"
    assert record["instrument"] == "endpoints"
    assert record["epochs_to_grok"]["epoch"] == 30
    assert record["epochs_to_grok"]["censored"] is False
    assert record["chance_accuracy"] == pytest.approx(1 / 8)
    assert record["leak_covariate"]["transpose_leak_fraction"] == 0.30
    assert record["accuracy"]["at_final_epoch"]["unleaked"] == pytest.approx(0.995)
    # Provenance pins the analysis code and the run's own hashes; no verdict.
    assert "endpoint_code_sha256" in record["provenance"]
    assert "endpoints.py" in record["provenance"]["endpoint_code_sha256"]
    assert record["provenance"]["checkpoint_sha256"]  # final.pt gated stable
    flat = " ".join(str(k) for k in record).lower()
    assert "verdict" not in flat and "label" not in flat


def test_measurement_vector_censored_run_is_still_measured(tmp_path):
    # Never sustains the bar: a censored seed still yields a full endpoint record.
    curve = [(e, 0.9 if e % 3 == 0 else 0.4) for e in range(200)]
    run = _write_run(tmp_path / "censored", unleaked_curve=curve)
    record = measurement_vector(run)
    assert record["status"] == "measured"
    assert record["epochs_to_grok"]["censored"] is True
    # No stable checkpoint for a never-grok run: recorded, not an error.
    assert record["checkpoint_selection"]["checkpoint"] is None


def test_measurement_vector_untrained_scores_near_chance_and_censors(tmp_path):
    # Rule-1 regression: an at-chance curve never groks.
    chance = 1 / 8
    curve = [(e, chance) for e in range(200)]
    run = _write_run(tmp_path / "untrained", unleaked_curve=curve, with_final_pt=False)
    record = measurement_vector(run)
    assert record["epochs_to_grok"]["censored"] is True
    assert record["accuracy"]["at_final_epoch"]["unleaked"] == pytest.approx(chance)


def test_measurement_vector_empty_unleaked_falls_back_to_raw(tmp_path):
    run = _write_run(
        tmp_path / "empty",
        unleaked_curve=[(e, float("nan")) for e in range(50)],
        unleaked_empty=True,
    )
    record = measurement_vector(run)
    assert record["endpoint_metric"] == "val/accuracy"
    assert record["endpoint_metric_is_fallback"] is True


def test_measurement_vector_skips_empty_log(tmp_path):
    run = _write_run(tmp_path / "nolog", unleaked_curve=_grok_curve(10, 50))
    (run / "run.log").write_text("startup banner, no epoch rows\n")
    record = measurement_vector(run)
    assert record["status"] == "skipped"


def test_measurement_vector_rows_without_endpoint_metric_is_skipped_not_censored(tmp_path):
    """Finding 1's second route: run.log rows exist but none carry the endpoint
    metric, so the grok series is empty. An empty series must not silently
    yield a measured-censored record -- indistinguishable from a genuine
    non-grokker once written -- it must be marked skipped instead."""
    run = _write_run(tmp_path / "nometric", unleaked_curve=_grok_curve(10, 50))
    lines = [f"epoch {e} | {{'train/loss': 0.01}}" for e in range(50)]
    (run / "run.log").write_text("\n".join(lines) + "\n")
    record = measurement_vector(run)
    assert record["status"] == "skipped"
    assert "val/unleaked_accuracy" in record["reason"]
    # The dip-aware checkpoint pick is still attached, as with the other skip path.
    assert "checkpoint_selection" in record


def test_measurement_vector_reads_gzipped_log(tmp_path):
    """A shipped run's log survives only as run.log.gz; the endpoint series is
    read straight from it, so the run measures fully."""
    import gzip

    run = _write_run(tmp_path / "gz", unleaked_curve=_grok_curve(onset=30, total=200))
    plain = run / "run.log"
    with gzip.open(run / "run.log.gz", "wt") as handle:
        handle.write(plain.read_text())
    plain.unlink()
    record = measurement_vector(run)
    assert record["status"] == "measured"
    assert record["epochs_to_grok"]["epoch"] == 30


def test_measurement_vector_no_log_still_carries_selection(tmp_path):
    """A curated run with no log at all degrades to skipped for the series-based
    endpoints, but still attaches the checkpoint pick recorded in
    selection.json rather than dropping the run entirely."""
    import json

    run = _write_run(tmp_path / "nolog2", unleaked_curve=_grok_curve(10, 50))
    (run / "run.log").unlink()
    (run / "final_epoch_49.pt").write_bytes(b"stub")  # flat curated checkpoint
    (run / "selection.json").write_text(
        json.dumps(
            {
                "selections": {
                    "stable_end": {
                        "rule": "final_gate",
                        "metric": "val/accuracy",
                        "threshold": 0.99,
                        "checkpoint": "final_epoch_49.pt",
                        "epoch": 49,
                        "metric_value": 1.0,
                        "substitution": None,
                        "rejected": [],
                        "reason": None,
                    }
                }
            }
        )
    )
    record = measurement_vector(run)
    assert record["status"] == "skipped"
    assert record["checkpoint_selection"]["checkpoint"] == "final_epoch_49.pt"
    assert record["checkpoint_selection"]["epoch"] == 49


def test_within_pair_endpoints_pairs_on_seed(tmp_path):
    a_dirs, b_dirs = [], []
    for seed in range(4):
        a_dirs.append(
            _write_run(
                tmp_path / f"a{seed}",
                seed=seed,
                index=3,
                unleaked_curve=_grok_curve(onset=50 + 10 * seed, total=200),
            )
        )
        b_dirs.append(
            _write_run(
                tmp_path / f"b{seed}",
                seed=seed,
                index=4,
                unleaked_curve=_grok_curve(onset=20 + 10 * seed, total=200),
            )
        )
    records_a = [measurement_vector(d) for d in a_dirs]
    records_b = [measurement_vector(d) for d in b_dirs]
    pair = within_pair_endpoints(records_a, records_b)
    assert pair["paired_seeds"] == [0, 1, 2, 3]
    grok_diff = pair["epochs_to_grok_difference"]
    # Every A onset is 30 epochs after the matched B onset.
    assert grok_diff["numeric_both_grokked"]["mean_difference"] == pytest.approx(30.0)
    assert grok_diff["sign_test"]["n_positive"] == 4.0


def test_within_pair_endpoints_rejects_duplicate_seed(tmp_path):
    # The corpus has same-seed reruns; a member with two measured records for
    # the same seed must raise rather than silently collapsing to one via a
    # dict comprehension.
    a0 = _write_run(tmp_path / "a0", seed=0, unleaked_curve=_grok_curve(30, 200))
    a0_rerun = _write_run(tmp_path / "a0_rerun", seed=0, unleaked_curve=_grok_curve(40, 200))
    b0 = _write_run(tmp_path / "b0", seed=0, unleaked_curve=_grok_curve(20, 200))
    records_a = [measurement_vector(a0), measurement_vector(a0_rerun)]
    records_b = [measurement_vector(b0)]
    with pytest.raises(ValueError, match="duplicate seeds"):
        within_pair_endpoints(records_a, records_b)


def test_within_pair_endpoints_excludes_metric_mismatched_seed_from_both_arms(tmp_path):
    # Seed 0: both members on the normal unleaked metric. Seed 1: member A's
    # unleaked subset was empty for that seed (a per-seed fallback), so it used
    # raw accuracy while member B used unleaked -- not a like-for-like
    # comparison on either arm, so seed 1 must be excluded from both rather
    # than silently comparing raw to unleaked (or reading a fallback member's
    # NaN unleaked accuracy).
    a0 = _write_run(tmp_path / "a0", seed=0, unleaked_curve=_grok_curve(30, 200))
    b0 = _write_run(tmp_path / "b0", seed=0, unleaked_curve=_grok_curve(20, 200))
    a1 = _write_run(
        tmp_path / "a1",
        seed=1,
        unleaked_curve=[(e, float("nan")) for e in range(200)],
        unleaked_empty=True,
    )
    b1 = _write_run(tmp_path / "b1", seed=1, unleaked_curve=_grok_curve(25, 200))

    records_a = [measurement_vector(a0), measurement_vector(a1)]
    records_b = [measurement_vector(b0), measurement_vector(b1)]
    assert records_a[1]["endpoint_metric_is_fallback"] is True

    pair = within_pair_endpoints(records_a, records_b)
    assert pair["paired_seeds"] == [0, 1]
    assert pair["metric_mismatched_seeds"] == [1]
    # Both arms computed over seed 0 only.
    assert pair["epochs_to_grok_difference"]["n_pairs"] == 1
    assert pair["final_unleaked_accuracy_difference"]["n_pairs"] == 1


def test_within_pair_endpoints_reports_seeds_skipped_in_both_members(tmp_path):
    a0 = _write_run(tmp_path / "a0", seed=0, unleaked_curve=_grok_curve(30, 200))
    b0 = _write_run(tmp_path / "b0", seed=0, unleaked_curve=_grok_curve(20, 200))
    # Seed 1 exists on both sides but neither run kept a log -- skipped on
    # both members, so it must show up in dropped_unmatched_seeds rather than
    # vanishing (the seed is absent from *both* by_seed dicts, so the plain
    # set differences that catch a_only/b_only never see it).
    a1 = _write_run(tmp_path / "a1", seed=1, unleaked_curve=_grok_curve(30, 200))
    (a1 / "run.log").write_text("startup banner, no epoch rows\n")
    b1 = _write_run(tmp_path / "b1", seed=1, unleaked_curve=_grok_curve(20, 200))
    (b1 / "run.log").write_text("startup banner, no epoch rows\n")

    records_a = [measurement_vector(a0), measurement_vector(a1)]
    records_b = [measurement_vector(b0), measurement_vector(b1)]
    assert records_a[1]["status"] == "skipped"
    assert records_b[1]["status"] == "skipped"

    pair = within_pair_endpoints(records_a, records_b)
    assert pair["paired_seeds"] == [0]
    assert pair["dropped_unmatched_seeds"]["skipped_in_both"] == [1]
    assert pair["dropped_unmatched_seeds"]["a_only"] == []
    assert pair["dropped_unmatched_seeds"]["b_only"] == []


# ---------------------------------------------------------------------------
# Salvaged v1 smoke tests (skipped when the archive is absent)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _ARCHIVE_GROKKED.is_dir(), reason="results-archive/ not present")
def test_archive_grokked_run_onset():
    series = metric_series(_ARCHIVE_GROKKED)
    grok = epochs_to_grok(series, ceiling=30000, threshold=0.99, sustain=5)
    assert grok.censored is False
    assert grok.epoch == 156  # hand-verified against this run's run.log


@pytest.mark.skipif(not _ARCHIVE_CENSORED.is_dir(), reason="results-archive/ not present")
def test_archive_censored_run_is_censored():
    record = measurement_vector(_ARCHIVE_CENSORED)
    assert record["status"] == "measured"
    assert record["epochs_to_grok"]["censored"] is True
    assert record["epochs_to_grok"]["ceiling"] == 30000
    assert record["accuracy"]["at_final_epoch"]["unleaked"] == pytest.approx(0.9189, abs=1e-3)
    assert not math.isnan(record["accuracy"]["at_final_epoch"]["unleaked"])


# ---------------------------------------------------------------------------
# scripts/measure_endpoints.py: the entry point itself, exercised end-to-end
# ---------------------------------------------------------------------------


def _load_measure_endpoints_script():
    """The script is an executable entry point, not part of the installed
    package, so it is loaded by file path (mirroring tests/test_preflight.py)."""
    import importlib.util
    import sys
    from types import ModuleType

    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "measure_endpoints_script", root / "scripts" / "measure_endpoints.py"
    )
    assert spec is not None and spec.loader is not None
    module: ModuleType = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_measure_endpoints_script_measure_and_pair_end_to_end(tmp_path):
    """No prior test in this suite ever runs scripts/measure_endpoints.py
    itself; this drives both subcommands against synthesised run dirs and
    checks the exit codes and the JSON each one writes."""
    import json

    script = _load_measure_endpoints_script()

    a_dirs, b_dirs = [], []
    for seed_val in range(3):
        a_dirs.append(
            _write_run(
                tmp_path / f"a{seed_val}",
                seed=seed_val,
                index=3,
                unleaked_curve=_grok_curve(onset=30 + 5 * seed_val, total=200),
            )
        )
        b_dirs.append(
            _write_run(
                tmp_path / f"b{seed_val}",
                seed=seed_val,
                index=4,
                unleaked_curve=_grok_curve(onset=20 + 5 * seed_val, total=200),
            )
        )

    measure_out = tmp_path / "measure.json"
    rc = script.main(
        [
            "measure",
            *[str(d) for d in a_dirs],
            *[str(d) for d in b_dirs],
            "--out",
            str(measure_out),
        ]
    )
    assert rc == 0
    payload = json.loads(measure_out.read_text())
    assert len(payload["runs"]) == 6
    assert all(r["status"] == "measured" for r in payload["runs"])
    for d in a_dirs + b_dirs:  # each run also writes its own analysis/endpoints.json
        assert (d / "analysis" / "endpoints.json").is_file()

    pair_out = tmp_path / "pair.json"
    rc = script.main(
        [
            "pair",
            "--member-a",
            *[str(d) for d in a_dirs],
            "--member-b",
            *[str(d) for d in b_dirs],
            "--out",
            str(pair_out),
        ]
    )
    assert rc == 0
    pair_record = json.loads(pair_out.read_text())
    assert pair_record["instrument"] == "endpoints-pair"
    assert pair_record["paired_seeds"] == [0, 1, 2]


def test_measure_endpoints_script_measure_exits_1_on_skip(tmp_path):
    script = _load_measure_endpoints_script()
    good = _write_run(tmp_path / "good", seed=0, unleaked_curve=_grok_curve(30, 200))
    bad = _write_run(tmp_path / "bad", seed=1, unleaked_curve=_grok_curve(30, 200))
    (bad / "run.log").write_text("startup banner, no epoch rows\n")
    rc = script.main(["measure", str(good), str(bad)])
    assert rc == 1


def test_measure_endpoints_script_pair_exits_1_when_a_member_run_is_skipped(tmp_path):
    """Finding 5: the `pair` subcommand must not exit 0 when a member run was
    skipped -- aligning its exit-code contract with `measure`, which already
    exits 1 in this situation."""
    script = _load_measure_endpoints_script()
    a0 = _write_run(tmp_path / "a0", seed=0, unleaked_curve=_grok_curve(30, 200))
    a1 = _write_run(tmp_path / "a1", seed=1, unleaked_curve=_grok_curve(30, 200))
    (a1 / "run.log").write_text("startup banner, no epoch rows\n")
    b0 = _write_run(tmp_path / "b0", seed=0, unleaked_curve=_grok_curve(20, 200))
    b1 = _write_run(tmp_path / "b1", seed=1, unleaked_curve=_grok_curve(20, 200))

    rc = script.main(
        [
            "pair",
            "--member-a",
            str(a0),
            str(a1),
            "--member-b",
            str(b0),
            str(b1),
        ]
    )
    assert rc == 1
