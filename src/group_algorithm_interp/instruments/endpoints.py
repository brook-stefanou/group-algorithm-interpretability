"""Tier-0 endpoint layer: the per-run measurement vector and the paired,
estimation-first estimators the study reports.

This module builds the endpoints that gate every downstream claim, reading
post-hoc from a run's committed artefacts (``run.log``, ``manifest.yaml``,
``resolved_config.yaml``) with no network, no W&B, and no re-training. It maps
onto the Tier-0 instruments of ``plan.md`` §8 as follows.

* **I-01** held-out generalisation anchor. Generalisation is defined here and
  nowhere else: the transpose-unleaked held-out accuracy series
  (``val/unleaked_accuracy`` in ``run.log``) and the epochs-to-grok onset
  computed from it (:func:`epochs_to_grok`). The chance level an untrained
  model sits at is ``1 / |G|`` (:func:`chance_accuracy`), reported so a reader
  can see the bar is a real bar.
* **I-02** leak-aware realised-leak covariate. ``task.py`` already exposes the
  transpose-unleaked subset and its realised leak fraction, and the training
  paths already record them into ``manifest.yaml`` under ``dataset.leakage``.
  :func:`realised_leak_covariate` reads that back as a per-run covariate and
  states the analytic identity ``leak ≈ train_frac × k(G)/|G|`` alongside
  the realised value. It computes; it does not change training or evaluation.
* **I-03** the null battery (endpoint-layer share). The accuracy endpoints are
  anchored on the chance level (:func:`chance_accuracy`); the paired estimators
  carry a paired sign-flip permutation as their exchangeability null
  (:func:`paired_sign_flip_permutation`). The model-level nulls (random-init,
  random-subspace, permuted-neuron) live with the instruments that consume
  them (``occupancy.py``, ``interventions.py``).
* **I-04** paired seed-replication and effect-size engine. :func:`paired_difference`
  for continuous endpoints and :func:`paired_grok_difference` for the
  censored epochs endpoint. Both resample the unit the claim is about — the
  seed within a pair — and report an effect size with a bootstrap confidence
  interval plus descriptive supplements (sign test, sign-flip permutation).
  No significance threshold, no equivalence margin: the interval is the
  statement.
* **I-06** signal-vector reporter. :func:`measurement_vector` assembles one
  run's full vector of endpoint signals with its provenance and no verdict
  label; :func:`within_pair_endpoints` runs I-04 over two members' vectors to
  produce the pair-level estimates the writeup reports.

Estimation-first discipline throughout: every measurement is reported, p-values
are descriptive supplements to an interval and never a gate, and no function
collapses its signals into a label.

The censoring rule is pre-registered (``docs/core-study.md`` §"Endpoints"): a
seed that never sustains ``> threshold`` unleaked accuracy for ``sustain``
consecutive evaluated epochs before the ceiling is censored. The ceiling is the
run's own ``optim.epochs`` (60,000 for the 2026-07-20 restart, 30,000 for the
salvaged v1 runs), read per run rather than hard-coded, so this instrument
tracks the config a run actually used.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .. import stats
from ..config import validate_config
from ..manifest import get_git_commit, read_manifest
from .checkpoints import parse_run_log, select_checkpoint
from .nulls import chance_accuracy

UNLEAKED_METRIC = "val/unleaked_accuracy"
RAW_METRIC = "val/accuracy"
LOSS_METRIC = "val/loss"
GROK_THRESHOLD = 0.99
GROK_SUSTAIN = 5


# ---------------------------------------------------------------------------
# Provenance helpers (mirrors instruments/report.py so records trace to code)
# ---------------------------------------------------------------------------


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def endpoint_code_hashes() -> dict[str, str]:
    """sha256 of the modules that compute the endpoint numbers, so a record
    pins the exact analysis code that produced it. Covers this module, the
    checkpoint selector it reuses, the shared null battery it draws the chance
    anchor from, and ``stats.py``; ``interventions.py`` is hashed too when
    present."""
    here = Path(__file__).resolve()
    package_dir = here.parent
    src_root = package_dir.parent
    candidates = [
        here,
        package_dir / "checkpoints.py",
        package_dir / "interventions.py",
        package_dir / "nulls.py",
        src_root / "stats.py",
    ]
    return {path.name: _file_sha256(path) for path in candidates if path.is_file()}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# I-01: held-out generalisation series and epochs-to-grok
# ---------------------------------------------------------------------------


def metric_series(run_dir: Path, metric: str = UNLEAKED_METRIC) -> list[tuple[int, float]]:
    """The ``(epoch, value)`` series for ``metric`` from a run's ``run.log``, in
    ascending epoch order. Reuses the shared ``run.log`` parser; rows with no
    entry for ``metric`` are skipped. Non-finite values are passed through as-is
    (a degenerate run records NaN unleaked accuracy) so the grok logic can treat
    them as failing the bar."""
    rows = parse_run_log(run_dir / "run.log")
    return [(row.epoch, row.metrics[metric]) for row in rows if metric in row.metrics]


@dataclass(frozen=True)
class GrokTime:
    """Epochs-to-grok for one run, with censoring made explicit.

    ``epoch`` is the onset epoch — the first epoch of the earliest streak of
    ``sustain`` consecutive evaluated rows above the bar — or ``None`` when the
    run is censored. ``censored`` is ``True`` iff no such streak occurred before
    the ceiling. ``ceiling`` is the run's own ``optim.epochs``. ``reached_ceiling``
    records whether the log actually reached the ceiling: a censored run whose
    log stops early (a crash, not a genuine non-grok) is flagged, not silently
    treated as a clean non-grokker.
    """

    epoch: int | None
    censored: bool
    ceiling: int
    metric: str
    threshold: float
    sustain: int
    grok_value: float | None
    n_evaluated_rows: int
    max_epoch_recorded: int | None
    reached_ceiling: bool

    def to_record(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "censored": self.censored,
            "ceiling": self.ceiling,
            "metric": self.metric,
            "threshold": self.threshold,
            "sustain": self.sustain,
            "grok_value": self.grok_value,
            "n_evaluated_rows": self.n_evaluated_rows,
            "max_epoch_recorded": self.max_epoch_recorded,
            "reached_ceiling": self.reached_ceiling,
        }


def epochs_to_grok(
    series: list[tuple[int, float]],
    *,
    ceiling: int,
    threshold: float = GROK_THRESHOLD,
    sustain: int = GROK_SUSTAIN,
    metric: str = UNLEAKED_METRIC,
) -> GrokTime:
    """Endpoint 1, pre-registered verbatim: the first epoch beginning a run of
    ``> threshold`` accuracy sustained for ``sustain`` or more consecutive
    evaluated epochs. A seed that never sustains the bar before ``ceiling`` is
    censored.

    ``series`` is the ``(epoch, value)`` list from :func:`metric_series` — the
    recorded evaluations, which the campaign writes every epoch (``log_every: 1``),
    so "consecutive evaluated rows" equals "consecutive epochs" there. The bar is
    strict (``value > threshold``), matching the ">0.99" wording; a NaN value
    (empty unleaked subset) never clears it. The reported onset is the epoch of
    the first row in the qualifying streak.
    """
    if sustain < 1:
        raise ValueError(f"sustain must be >= 1, got {sustain}")
    ordered = sorted(series)
    n = len(ordered)
    max_epoch = ordered[-1][0] if ordered else None
    reached_ceiling = max_epoch is not None and max_epoch >= ceiling - 1
    streak = 0
    onset_epoch: int | None = None
    for epoch, value in ordered:
        if math.isfinite(value) and value > threshold:
            if streak == 0:
                candidate = epoch
            streak += 1
            if streak >= sustain:
                onset_epoch = candidate
                break
        else:
            streak = 0
    if onset_epoch is not None:
        grok_value = next(v for e, v in ordered if e == onset_epoch)
        return GrokTime(
            epoch=onset_epoch,
            censored=False,
            ceiling=ceiling,
            metric=metric,
            threshold=threshold,
            sustain=sustain,
            grok_value=grok_value,
            n_evaluated_rows=n,
            max_epoch_recorded=max_epoch,
            reached_ceiling=reached_ceiling,
        )
    return GrokTime(
        epoch=None,
        censored=True,
        ceiling=ceiling,
        metric=metric,
        threshold=threshold,
        sustain=sustain,
        grok_value=None,
        n_evaluated_rows=n,
        max_epoch_recorded=max_epoch,
        reached_ceiling=reached_ceiling,
    )


# ---------------------------------------------------------------------------
# I-02: leak-aware realised-leak covariate (read from the manifest)
# ---------------------------------------------------------------------------


def realised_leak_covariate(manifest: dict[str, Any], train_frac: float | None) -> dict[str, Any]:
    """I-02 as a pure covariate read: the realised transpose-leak fraction and
    the quantities the analytic estimate ``train_frac × k(G)/|G|`` is built
    from, taken from what the training path already recorded under
    ``manifest['dataset']['leakage']``. Nothing here changes training or
    evaluation.

    ``k(G)/|G|`` is the group's commuting probability (``task.commuting_probability``);
    the training path stored it as ``commuting_probability``. The estimate and
    the realised value differ because the split is a finite random draw — the
    residual is reported so the covariate carries its own calibration.
    """
    leakage = (manifest.get("dataset") or {}).get("leakage") or {}
    realised = leakage.get("transpose_leak_fraction")
    commuting = leakage.get("commuting_probability")
    estimate: float | None = None
    residual: float | None = None
    if train_frac is not None and commuting is not None:
        estimate = float(train_frac) * float(commuting)
        if realised is not None:
            residual = float(realised) - estimate
    return {
        "transpose_leak_fraction": realised,
        "commuting_probability": commuting,
        "train_frac": train_frac,
        "leak_estimate_trainfrac_times_commuting": estimate,
        "realised_minus_estimate": residual,
        "test_size": leakage.get("test_size"),
        "unleaked_test_size": leakage.get("unleaked_test_size"),
        "unleaked_empty": leakage.get("unleaked_empty"),
        "generalize_metric": leakage.get("generalize_metric"),
    }


# ---------------------------------------------------------------------------
# I-04: paired estimators (continuous, and censored epochs)
# ---------------------------------------------------------------------------


def paired_sign_flip_permutation(
    diffs: list[float],
    *,
    n_resamples: int = 10_000,
    seed: int = 0,
    exact_max_n: int = 22,
) -> dict[str, float]:
    """Paired sign-flip permutation test over per-pair differences — the
    descriptive supplement §"Endpoints" allows alongside an interval.

    The exchangeability null is that each pair's sign is arbitrary, so flipping
    the sign of any subset of the ``diffs`` is equally likely under the null.
    The statistic is ``|mean(diffs)|``; the p-value is the fraction of the
    ``2**n`` sign assignments whose ``|mean|`` is at least the observed one. When
    ``n <= exact_max_n`` every assignment is enumerated (deterministic, needs no
    seed); above that, ``n_resamples`` assignments are drawn with a private
    seeded RNG so the global streams a run depends on are untouched. The floor is
    ``1 / 2**n`` (exact, the observed assignment is always included) or
    ``1 / (n_resamples + 1)`` (sampled). Returns
    ``{statistic, p_two_sided, n_permutations, exact}``; a p-value here never
    gates a claim.
    """
    vals = [float(d) for d in diffs]
    n = len(vals)
    if n == 0:
        raise ValueError("need at least one paired difference for a sign-flip permutation")
    observed = abs(statistics.fmean(vals))
    if n <= exact_max_n:
        as_extreme = 0
        for signs in itertools.product((1.0, -1.0), repeat=n):
            flipped = statistics.fmean([s * v for s, v in zip(signs, vals, strict=True)])
            if abs(flipped) >= observed:
                as_extreme += 1
        total = 2**n
        return {
            "statistic": observed,
            "p_two_sided": min(1.0, as_extreme / total),
            "n_permutations": float(total),
            "exact": 1.0,
        }
    import random

    rng = random.Random(seed)
    as_extreme = 0
    for _ in range(n_resamples):
        flipped = statistics.fmean([(1.0 if rng.random() < 0.5 else -1.0) * v for v in vals])
        if abs(flipped) >= observed:
            as_extreme += 1
    return {
        "statistic": observed,
        "p_two_sided": min(1.0, (as_extreme + 1) / (n_resamples + 1)),
        "n_permutations": float(n_resamples),
        "exact": 0.0,
    }


def paired_difference(
    a_values: list[float],
    b_values: list[float],
    *,
    label: str = "a_minus_b",
    units: str | None = None,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """I-04 for a continuous endpoint (e.g. transpose-unleaked accuracy): the
    per-seed paired difference ``a - b``, an effect size, and a paired bootstrap
    confidence interval, with a sign test and a sign-flip permutation as
    descriptive supplements.

    ``a_values`` and ``b_values`` are the same seeds' measurements on the two
    members, aligned by position (the caller pairs by seed). The bootstrap
    resamples the per-pair differences — the seed within the pair, which is the
    unit the pair-level claim is about — so it never pseudoreplicates. The
    interval is the statement; the p-values are supplements and gate nothing.
    """
    if len(a_values) != len(b_values):
        raise ValueError(f"unequal seed counts: {len(a_values)} vs {len(b_values)}")
    if not a_values:
        raise ValueError("need at least one paired observation")
    diffs = [float(a) - float(b) for a, b in zip(a_values, b_values, strict=True)]
    n = len(diffs)
    mean = statistics.fmean(diffs)
    median = statistics.median(diffs)
    std = statistics.stdev(diffs) if n > 1 else 0.0
    dz = mean / std if std > 0 else None
    ci: list[float] | None
    if n >= 2:
        low, high = stats.bootstrap_ci(
            diffs, confidence=confidence, n_resamples=n_resamples, seed=seed
        )
        ci = [low, high]
    else:
        ci = None
    return {
        "label": label,
        "units": units,
        "n_pairs": n,
        "per_seed_difference": diffs,
        "mean_difference": mean,
        "median_difference": median,
        "std_difference": std,
        "standardised_effect_dz": dz,
        "bootstrap_ci_95": ci,
        "sign_test": stats.sign_test(diffs),
        "sign_flip_permutation": paired_sign_flip_permutation(diffs, seed=seed),
    }


def paired_grok_difference(
    a_times: list[GrokTime],
    b_times: list[GrokTime],
    *,
    label: str = "a_minus_b",
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """I-04 for the censored epochs-to-grok endpoint, applying the pre-registered
    censoring rule to paired seeds.

    For each seed pair, ``a - b`` in epochs:

    * both members grokked — the numeric difference, which feeds the effect
      size and the bootstrap interval;
    * one member censored — the censored member is at least the ceiling and the
      other is below it, so the sign is known (the censored member is slower);
      the pair keeps its sign for the sign test but is left out of the numeric
      interval, whose magnitude it would only bound;
    * both censored — a tie, dropped from the sign test.

    The censoring fraction of each member is reported as its own measurement. The
    numeric interval covers the both-grokked subset only, with the excluded count
    stated so the reader sees how much of the pair is censored rather than
    estimated. Units are epochs. No verdict, no threshold.
    """
    if len(a_times) != len(b_times):
        raise ValueError(f"unequal seed counts: {len(a_times)} vs {len(b_times)}")
    if not a_times:
        raise ValueError("need at least one paired observation")

    signed_for_sign_test: list[float] = []
    finite_diffs: list[float] = []
    n_both_finite = 0
    n_a_censored_only = 0
    n_b_censored_only = 0
    n_both_censored = 0
    for a, b in zip(a_times, b_times, strict=True):
        if not a.censored and not b.censored:
            assert a.epoch is not None and b.epoch is not None
            d = float(a.epoch - b.epoch)
            finite_diffs.append(d)
            signed_for_sign_test.append(d)
            n_both_finite += 1
        elif a.censored and not b.censored:
            # a is at least the ceiling, b grokked below it: a - b > 0.
            signed_for_sign_test.append(1.0)
            n_a_censored_only += 1
        elif b.censored and not a.censored:
            signed_for_sign_test.append(-1.0)
            n_b_censored_only += 1
        else:
            n_both_censored += 1

    n_pairs = len(a_times)
    numeric: dict[str, Any] = {
        "n_both_grokked": n_both_finite,
        "n_excluded_censored": n_pairs - n_both_finite,
        "per_seed_difference": finite_diffs,
    }
    if n_both_finite >= 1:
        numeric["mean_difference"] = statistics.fmean(finite_diffs)
        numeric["median_difference"] = statistics.median(finite_diffs)
        numeric["std_difference"] = statistics.stdev(finite_diffs) if n_both_finite > 1 else 0.0
    else:
        numeric["mean_difference"] = None
        numeric["median_difference"] = None
        numeric["std_difference"] = None
    if n_both_finite >= 2:
        low, high = stats.bootstrap_ci(
            finite_diffs, confidence=confidence, n_resamples=n_resamples, seed=seed
        )
        numeric["bootstrap_ci_95"] = [low, high]
        numeric["sign_flip_permutation"] = paired_sign_flip_permutation(finite_diffs, seed=seed)
    else:
        numeric["bootstrap_ci_95"] = None
        numeric["sign_flip_permutation"] = None

    ceiling = a_times[0].ceiling
    return {
        "label": label,
        "units": "epochs",
        "ceiling": ceiling,
        "n_pairs": n_pairs,
        "n_both_grokked": n_both_finite,
        "n_a_censored_only": n_a_censored_only,
        "n_b_censored_only": n_b_censored_only,
        "n_both_censored_ties": n_both_censored,
        "censoring_fraction_a": sum(1 for t in a_times if t.censored) / n_pairs,
        "censoring_fraction_b": sum(1 for t in b_times if t.censored) / n_pairs,
        # Censoring-robust: signs are well-defined for every non-both-censored
        # pair, so the sign test uses them all; both-censored ties are dropped.
        "sign_test": stats.sign_test(signed_for_sign_test),
        "numeric_both_grokked": numeric,
    }


# ---------------------------------------------------------------------------
# I-06: the per-run signal vector
# ---------------------------------------------------------------------------


def _row_value(rows: dict[int, dict[str, float]], epoch: int | None, metric: str) -> float | None:
    if epoch is None:
        return None
    metrics = rows.get(epoch)
    if metrics is None or metric not in metrics:
        return None
    return metrics[metric]


def measurement_vector(
    run_dir: Path,
    *,
    threshold: float = GROK_THRESHOLD,
    sustain: int = GROK_SUSTAIN,
) -> dict[str, Any]:
    """I-06: one run's full vector of endpoint signals with provenance and no
    verdict label.

    The record carries, per run: the epochs-to-grok endpoint (censored or not),
    the transpose-unleaked and raw held-out accuracy at both the dip-aware
    checkpoint and the final recorded epoch, the chance-accuracy anchor, the
    I-02 realised-leak covariate, and the dip-aware checkpoint selection (which
    may legitimately find no stable checkpoint for a censored seed — recorded as
    data, not an error). A run whose ``run.log`` has no metric rows is returned
    with ``status: "skipped"``.

    Grok is measured on the transpose-unleaked subset. In the degenerate case
    where that subset is empty (``unleaked_empty``) the run's own training used
    raw accuracy as the fallback bar; this instrument mirrors that, records the
    fallback metric, and flags it, so the endpoint is never silently computed on
    NaN.
    """
    manifest = read_manifest(run_dir)
    config = validate_config(yaml.safe_load((run_dir / "resolved_config.yaml").read_text()))
    ceiling = config.optim.epochs
    train_frac = config.data.train_frac

    leak = realised_leak_covariate(manifest, train_frac)
    endpoint_metric = UNLEAKED_METRIC
    metric_fallback = False
    if leak.get("unleaked_empty") is True:
        endpoint_metric = RAW_METRIC
        metric_fallback = True

    record: dict[str, Any] = {
        "instrument": "endpoints",
        "run_id": manifest.get("run_id", run_dir.name),
        "seed": config.seed,
        "group": {
            "order": config.data.group.order,
            "index": config.data.group.index,
            "name": config.data.group.canonical_name,
        },
        "model": {
            "arch": config.model.arch,
            "d_model": config.model.d_model,
            "d_mlp": config.model.d_mlp,
            "activation": config.model.activation,
        },
        "endpoint_metric": endpoint_metric,
        "endpoint_metric_is_fallback": metric_fallback,
        "leak_covariate": leak,
        "chance_accuracy": chance_accuracy(config.data.group.order),
        "provenance": {
            "git_commit": (manifest.get("provenance") or {}).get("git_commit"),
            "config_hash": (manifest.get("provenance") or {}).get("config_hash"),
            "config_group_hash": (manifest.get("provenance") or {}).get("config_group_hash"),
            "campaign_id": (manifest.get("provenance") or {}).get("campaign_id"),
            "dataset_spec_hash": (manifest.get("dataset") or {}).get("spec_hash"),
            "analysis_git_commit": get_git_commit(),
            "analysed_at": _utcnow(),
            "endpoint_code_sha256": endpoint_code_hashes(),
        },
        "note": (
            "Estimation-first measurement vector (I-06): endpoint signals with "
            "provenance and no verdict label. Grok is the onset of a sustained "
            "run above the unleaked bar; a censored seed carries no epoch."
        ),
    }

    rows_list = parse_run_log(run_dir / "run.log") if (run_dir / "run.log").is_file() else []
    if not rows_list:
        record["status"] = "skipped"
        record["reason"] = "run.log absent or has no epoch metric rows"
        return record
    rows = {row.epoch: row.metrics for row in rows_list}

    series = [
        (row.epoch, row.metrics[endpoint_metric])
        for row in rows_list
        if endpoint_metric in row.metrics
    ]
    grok = epochs_to_grok(
        series, ceiling=ceiling, threshold=threshold, sustain=sustain, metric=endpoint_metric
    )

    # Dip-aware checkpoint on the endpoint metric (falls back to raw for an
    # empty unleaked subset, matching the endpoint metric).
    selection = select_checkpoint(run_dir, metric=endpoint_metric, threshold=threshold)
    final_epoch = max(rows)

    record["status"] = "measured"
    record["epochs_to_grok"] = grok.to_record()
    record["checkpoint_selection"] = selection.to_record()
    record["accuracy"] = {
        "at_selected_checkpoint": {
            "epoch": selection.epoch,
            "unleaked": _row_value(rows, selection.epoch, UNLEAKED_METRIC),
            "raw": _row_value(rows, selection.epoch, RAW_METRIC),
            "loss": _row_value(rows, selection.epoch, LOSS_METRIC),
        },
        "at_final_epoch": {
            "epoch": final_epoch,
            "unleaked": _row_value(rows, final_epoch, UNLEAKED_METRIC),
            "raw": _row_value(rows, final_epoch, RAW_METRIC),
            "loss": _row_value(rows, final_epoch, LOSS_METRIC),
        },
    }
    if selection.path is not None:
        record["provenance"]["checkpoint_sha256"] = _file_sha256(selection.path)
    return record


def within_pair_endpoints(
    records_a: list[dict[str, Any]],
    records_b: list[dict[str, Any]],
    *,
    seed: int = 0,
) -> dict[str, Any]:
    """Pair-level I-04 estimates from two members' I-06 vectors: the paired
    epochs-to-grok difference (censored) and the paired final-unleaked-accuracy
    difference, over the seeds present in both members.

    Seeds are matched on the ``seed`` field; a seed present on only one member is
    dropped to preserve the pairing (banned practice 6). Both members must be
    measured records. The aggregation unit is the pair; the CIs are paired
    bootstraps over the matched seeds.
    """
    by_seed_a = {r["seed"]: r for r in records_a if r.get("status") == "measured"}
    by_seed_b = {r["seed"]: r for r in records_b if r.get("status") == "measured"}
    shared = sorted(set(by_seed_a) & set(by_seed_b))
    dropped_a = sorted(set(by_seed_a) - set(by_seed_b))
    dropped_b = sorted(set(by_seed_b) - set(by_seed_a))
    if not shared:
        raise ValueError("the two members share no measured seed; cannot pair")

    def _grok(record: dict[str, Any]) -> GrokTime:
        g = record["epochs_to_grok"]
        return GrokTime(
            epoch=g["epoch"],
            censored=g["censored"],
            ceiling=g["ceiling"],
            metric=g["metric"],
            threshold=g["threshold"],
            sustain=g["sustain"],
            grok_value=g["grok_value"],
            n_evaluated_rows=g["n_evaluated_rows"],
            max_epoch_recorded=g["max_epoch_recorded"],
            reached_ceiling=g["reached_ceiling"],
        )

    a_grok = [_grok(by_seed_a[s]) for s in shared]
    b_grok = [_grok(by_seed_b[s]) for s in shared]
    a_acc = [by_seed_a[s]["accuracy"]["at_final_epoch"]["unleaked"] for s in shared]
    b_acc = [by_seed_b[s]["accuracy"]["at_final_epoch"]["unleaked"] for s in shared]

    def _member(records: list[dict[str, Any]]) -> dict[str, Any]:
        first = records[0]
        return {
            "order": first["group"]["order"],
            "index": first["group"]["index"],
            "name": first["group"]["name"],
        }

    return {
        "instrument": "endpoints-pair",
        "member_a": _member(records_a),
        "member_b": _member(records_b),
        "paired_seeds": shared,
        "dropped_unmatched_seeds": {"a_only": dropped_a, "b_only": dropped_b},
        "epochs_to_grok_difference": paired_grok_difference(a_grok, b_grok, seed=seed),
        "final_unleaked_accuracy_difference": paired_difference(
            a_acc, b_acc, label="a_minus_b", units="unleaked accuracy", seed=seed
        ),
        "provenance": {
            "analysis_git_commit": get_git_commit(),
            "analysed_at": _utcnow(),
            "endpoint_code_sha256": endpoint_code_hashes(),
        },
        "note": (
            "Pair-level estimation-first endpoints (I-04 over I-06 vectors): "
            "effect size and paired bootstrap interval per pair, no verdict."
        ),
    }


__all__ = [
    "GROK_SUSTAIN",
    "GROK_THRESHOLD",
    "GrokTime",
    "LOSS_METRIC",
    "RAW_METRIC",
    "UNLEAKED_METRIC",
    "chance_accuracy",
    "endpoint_code_hashes",
    "epochs_to_grok",
    "measurement_vector",
    "metric_series",
    "paired_difference",
    "paired_grok_difference",
    "paired_sign_flip_permutation",
    "realised_leak_covariate",
    "within_pair_endpoints",
]
