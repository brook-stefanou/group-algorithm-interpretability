"""Small, zero-dependency statistics helpers.

Nothing in ``src/`` or ``scripts/`` imports this module yet: it is a bench of
instruments waiting for a caller, not (yet) part of any experiment's real
analysis path. Comparing two conditions across seeds is the workhorse of
experiment analysis, so this is where each stat lands as a claim needs it --
keep each one pure and covered by a test.

The estimation-first reporting discipline (``docs/methodology.md``) wants
Wilson/Clopper-Pearson intervals and a paired bootstrap for the
accuracy-comparison situations this project actually
has; none of those exist here yet. ``permutation_test`` below fills the gap
that mattered most in the meantime: comparing two groups' accuracies when they
saturate at exactly 0/1 across seeds, which is exactly where ``welch_ttest``
(correctly) refuses to run.

Welch's unequal-variance t-test also lives here in pure Python -- no scipy
dependency. The p-value comes from the Student-t survival function via the
regularised incomplete beta function (Lentz's continued fraction).
"""

from __future__ import annotations

import itertools
import math
import random
import statistics
from collections.abc import Callable, Sequence


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    """Sample mean and std (ddof=1) -- the multi-seed ``mean ± std`` you report.
    std is 0.0 for a single value; raises on an empty sequence."""
    if not values:
        raise ValueError("need at least one value")
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def bootstrap_ci(
    values: Sequence[float],
    *,
    statistic: Callable[[Sequence[float]], float] = statistics.fmean,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for ``statistic`` over ``values``.
    Uses a private seeded RNG so it is reproducible and never perturbs the global
    ``random``/NumPy/torch streams a run depends on. Raises on fewer than 2 values."""
    vals = list(values)
    if len(vals) < 2:
        raise ValueError("need at least 2 values for a bootstrap CI")
    rng = random.Random(seed)
    n = len(vals)
    stats_sorted = sorted(
        statistic([vals[rng.randrange(n)] for _ in range(n)]) for _ in range(n_resamples)
    )
    alpha = (1.0 - confidence) / 2.0
    lo = stats_sorted[int(alpha * n_resamples)]
    hi = stats_sorted[min(n_resamples - 1, int((1.0 - alpha) * n_resamples))]
    return lo, hi


def sign_test(diffs: Sequence[float], *, zero_tol: float = 0.0) -> dict[str, float]:
    """Two-sided paired sign test over per-pair differences. Ties (``|diff| <=
    zero_tol``) are dropped; the two-sided p-value is the exact binomial (p=0.5)
    over the sign counts. Returns ``{n, n_positive, p_two_sided}`` (n = non-ties)."""
    pos = sum(1 for d in diffs if d > zero_tol)
    neg = sum(1 for d in diffs if d < -zero_tol)
    n = pos + neg
    if n == 0:
        return {"n": 0.0, "n_positive": float(pos), "p_two_sided": 1.0}
    k = min(pos, neg)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5**n)
    return {"n": float(n), "n_positive": float(pos), "p_two_sided": min(1.0, 2.0 * tail)}


def permutation_test(
    a: Sequence[float],
    b: Sequence[float],
    *,
    statistic: Callable[[Sequence[float]], float] = statistics.fmean,
    n_resamples: int = 10_000,
    seed: int = 0,
    exact_threshold: int = 10_000,
) -> dict[str, float]:
    """Two-sided permutation test comparing ``statistic`` between two independent
    groups. This is the tool for exactly the situation ``welch_ttest`` refuses to
    run: comparing accuracies that saturate at 0/1 across seeds, where there is
    no within-group variance to build a t-statistic from.

    Null hypothesis: group membership does not matter, so the observed pooled
    values are exchangeable between the two groups. Observed statistic:
    ``abs(statistic(a) - statistic(b))`` (default ``statistic`` is the mean; pass
    something else, e.g. a median, if that is the claim). The p-value is the
    fraction of relabellings of the pooled data into groups of sizes
    ``len(a)``/``len(b)`` whose relabelled statistic is at least as extreme as the
    one observed -- this always includes the observed labeling itself, so the
    minimum achievable p-value is ``1 / n_permutations`` (exact) or
    ``1 / (n_resamples + 1)`` (sampled): a perfectly separated pair of groups
    gets the smallest p the resolution allows, never an error and never exactly
    zero.

    When the number of distinct relabellings (``C(len(a)+len(b), len(a))``) is at
    most ``exact_threshold``, every relabelling is enumerated exactly, which is
    deterministic and needs no seed. Above that, ``n_resamples`` relabellings are
    drawn with a private seeded RNG (``random.Random(seed)``), so this never
    perturbs the global ``random``/NumPy/torch streams a run depends on, and is
    reproducible for a fixed seed.

    Returns ``{statistic, p_two_sided, n_permutations, exact}`` where ``exact``
    is ``1.0`` if every relabelling was enumerated and ``0.0`` if sampled.
    Raises ``ValueError`` if either group is empty.
    """
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        raise ValueError("each group needs at least 1 observation for a permutation test")
    pooled = list(a) + list(b)
    n = na + nb
    observed = abs(statistic(pooled[:na]) - statistic(pooled[na:]))
    n_exact = math.comb(n, na)
    if n_exact <= exact_threshold:
        as_extreme = 0
        for combo in itertools.combinations(range(n), na):
            in_a = set(combo)
            group_a = [pooled[i] for i in combo]
            group_b = [pooled[i] for i in range(n) if i not in in_a]
            diff = abs(statistic(group_a) - statistic(group_b))
            if diff >= observed:
                as_extreme += 1
        return {
            "statistic": observed,
            "p_two_sided": min(1.0, as_extreme / n_exact),
            "n_permutations": float(n_exact),
            "exact": 1.0,
        }
    rng = random.Random(seed)
    idx = list(range(n))
    as_extreme = 0
    for _ in range(n_resamples):
        rng.shuffle(idx)
        group_a = [pooled[i] for i in idx[:na]]
        group_b = [pooled[i] for i in idx[na:]]
        diff = abs(statistic(group_a) - statistic(group_b))
        if diff >= observed:
            as_extreme += 1
    return {
        "statistic": observed,
        "p_two_sided": min(1.0, (as_extreme + 1) / (n_resamples + 1)),
        "n_permutations": float(n_resamples),
        "exact": 0.0,
    }


def _betacf(a: float, b: float, x: float, iters: int = 200) -> float:
    """Continued fraction for the incomplete beta (Lentz's algorithm)."""
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, iters + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return h


def betainc_reg(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _student_t_sf_two_sided(t: float, dof: float) -> float:
    """Two-sided p-value P(|T| > |t|) for a Student-t with ``dof`` degrees of freedom."""
    if dof <= 0:
        return float("nan")
    x = dof / (dof + t * t)
    return betainc_reg(dof / 2.0, 0.5, x)


def welch_ttest(a: list[float], b: list[float]) -> dict[str, float]:
    """Welch's unequal-variance t-test. Returns ``{t, dof, p_two_sided}`` where
    ``dof`` is the Welch-Satterthwaite degrees of freedom. Raises ``ValueError``
    if either group has fewer than 2 observations.

    Degenerate case -- both groups have zero within-group variance (every
    observation in ``a`` is identical, and likewise for ``b``; e.g. saturated
    accuracies of exactly 1.0 or 0.0 across seeds, which is the expected shape
    of a real grokking-style result). Then the pooled standard error is exactly
    zero, so the t statistic is 0/0 whenever the two group means also differ --
    the samples are in fact *perfectly separated*, and the t-test is the wrong
    instrument regardless of what number it returns (scipy reports t=+-inf,
    p=0.0, which is technically consistent but invites silently reporting a
    t-test result that should never have been run). This function raises
    instead, naming the situation and pointing at the appropriate alternative:
    report the separation directly, or use a permutation test or an exact test
    (e.g. Fisher's exact test on saturation counts) that does not require
    within-group variance.

    The one sub-case handled without raising is when the means are *also*
    equal (the two samples are literally identical constants): that is a
    genuine "no difference" input, not an instrument failure, so it returns
    t=0.0, p_two_sided=1.0 -- the two-sided p-value at t=0 is 1.0 for any valid
    (positive) dof, so this holds regardless of how dof is chosen. dof itself
    is reported as the limit of the Welch-Satterthwaite formula as both
    variances shrink to zero at the same rate (va == vb): that limit is finite
    and depends only on na, nb, because the W-S formula is homogeneous of
    degree 0 in the variances. For na == nb it collapses to the familiar
    pooled dof na+nb-2, which is why that was a reasonable-looking value for
    the old, incorrectly-triggered fallback.
    """
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        raise ValueError("each group needs at least 2 observations for a t-test")
    va, vb = statistics.variance(a), statistics.variance(b)
    mean_a, mean_b = statistics.fmean(a), statistics.fmean(b)
    if va == 0.0 and vb == 0.0:
        if mean_a != mean_b:
            raise ValueError(
                "welch_ttest: both groups have zero within-group variance but "
                f"different means ({mean_a!r} vs {mean_b!r}) -- the samples are "
                "perfectly separated and the t-test is degenerate (standard error "
                "is exactly zero, so t is undefined/infinite). A t-test is the "
                "wrong instrument for this input regardless of what number it "
                "returns. Report the separation directly, or use a permutation "
                "test or an exact test (e.g. Fisher's exact test on saturation "
                "counts) instead."
            )
        dof = (1.0 / na + 1.0 / nb) ** 2 / ((1.0 / na**2) / (na - 1) + (1.0 / nb**2) / (nb - 1))
        return {"t": 0.0, "dof": dof, "p_two_sided": 1.0}
    sa, sb = va / na, vb / nb
    se2 = sa + sb
    t = (mean_a - mean_b) / math.sqrt(se2)
    dof = se2**2 / (sa**2 / (na - 1) + sb**2 / (nb - 1))
    return {"t": t, "dof": dof, "p_two_sided": _student_t_sf_two_sided(t, dof)}
