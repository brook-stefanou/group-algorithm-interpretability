#!/usr/bin/env python3
"""Within-pair contrasts across the character-table-equivalent panel -- the
committed derivation behind the portfolio draft's matched-pair numbers.

For each of the seven newer corpus instruments (occupancy, recruited
dimension, used-set audit sufficiency and necessity, gcr_matmul,
readout_characterisation, gcr_readout, coset_quotient), and for every pair of
groups in ``PAIRS`` below with committed per-cell results for BOTH members at
a given width, this script:

1. Extracts that instrument's headline scalar per seed (``HEADLINE_FIELDS``
   documents the exact field path for each).
2. Pairs the two members' seeds by seed number and takes the per-seed
   difference (member A minus member B).
3. Reports the mean difference with a paired bootstrap 95% CI
   (``group_algorithm_interp.stats.bootstrap_ci``, seed 0 -- the project's
   fixed convention for a paired-seed contrast).
4. Runs a two-sided paired sign-flip permutation test on the same per-seed
   differences (swapping which member is "A" vs "B" for a seed flips that
   seed's diff, so this is exactly the paired-relabelling null): exact
   enumeration of all ``2**n`` sign patterns for ``n <= --exact-max-n``, else
   Monte Carlo with a private seeded RNG.

Because every pair here is, by construction, identical on the invariants the
Fourier/tensor-rank/coset accounts read (see ``docs/core-study.md``'s C1), a
within-pair contrast's null hypothesis is that the paired difference is
exactly zero; a confidence interval or permutation p-value that excludes
zero is evidence against that joint prediction. Running the same test on
every pair in every instrument multiplies the number of independent looks,
so this script also runs a Benjamini-Hochberg step-up procedure (q = 0.05 and
0.10) over every computed contrast's permutation p-value in one pass, and
reports a binomial check of the observed zero-clearing count against a naive
5%-per-look null.

Pair categories (``PairSpec.category``):

- ``c1_matched``: the six genuine-clean, character-table-equivalent C1
  tier-1 pairs (``results/falsifier_screen_results_full.json``'s
  ``genuine_clean_pairs``) that have committed per-cell results for both
  members at some width. Any nonzero contrast here falsifies the joint
  Fourier/tensor-rank/coset prediction of no within-pair difference.
- ``c1_matched_staged``: the five remaining tier-1 pairs, still behind the
  pre-registered width-256+ staging gate (``docs/core-study.md``) -- no
  per-cell data exists for these yet, so they are listed but never
  contribute a contrast.
- ``c4_tier2_coset_confounded``: ``(216,106)/(216,107)``, FS-identical but
  not coset-clean -- a difference here still falsifies pure-Fourier, but its
  attribution is shared between the power-map and coset axes.
- ``fs_flip_control``: pairs that share an ordinary character table but
  differ in Frobenius-Schur indicators (``(64,60)/(64,65)``,
  ``(104,4)/(104,6)``, and the C2 case-study pair ``(32,18)/(32,20)``,
  D32/Q32). A difference here is *expected by design*, not a falsifier --
  these are direction tests / the C2 mechanism case study, included as a
  control on the multiple-looks battery.
- ``not_ct_equal_control``: ``(48,28)/(48,29)`` (C5) and
  ``(192,10)/(192,24)`` (E8) do not even share a character table (confirmed
  against the falsifier screen: neither pair shares a
  ``character_table_fingerprint`` bucket), so they carry zero C1
  falsification weight. Included only as a further multiple-looks control.

Determinism: every number derives from the committed ``results/*.json`` per-
cell files plus the fixed bootstrap/permutation seeds below; the output is
byte-reproducible given the same inputs. Input file inventory (path -> sha256)
is recorded in ``provenance.input_files_sha256`` so a change in any consumed
per-cell file is detectable without re-running anything.

Usage::

    uv run python scripts/derive_within_pair_contrasts.py
    uv run python scripts/derive_within_pair_contrasts.py --output results/within_pair_contrasts/within_pair_contrasts.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any, NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from group_algorithm_interp import stats  # noqa: E402
from group_algorithm_interp.manifest import get_git_commit, get_git_dirty  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
DEFAULT_OUTPUT = RESULTS / "within_pair_contrasts" / "within_pair_contrasts.json"

GroupId = tuple[int, int]


class PairSpec(NamedTuple):
    a: GroupId
    b: GroupId
    category: str
    label: str


#: The panel this script screens. See the module docstring for what each
#: category means and which account(s) a nonzero contrast bears on.
PAIRS: list[PairSpec] = [
    PairSpec((27, 3), (27, 4), "c1_matched", "C1 tier-1 (genuine clean)"),
    PairSpec((54, 10), (54, 11), "c1_matched", "C1 tier-1 embedding probe (genuine clean)"),
    PairSpec((64, 74), (64, 80), "c1_matched", "C1 tier-1 (genuine clean)"),
    PairSpec((64, 228), (64, 229), "c1_matched", "C1 tier-1 (genuine clean)"),
    PairSpec((64, 236), (64, 240), "c1_matched", "C1 tier-1 (genuine clean)"),
    PairSpec((64, 241), (64, 242), "c1_matched", "C1 tier-1 (genuine clean)"),
    PairSpec((81, 12), (81, 13), "c1_matched_staged", "C1 tier-1 (genuine clean, staged)"),
    PairSpec((125, 3), (125, 4), "c1_matched_staged", "C1 tier-1 (genuine clean, staged)"),
    PairSpec((243, 56), (243, 57), "c1_matched_staged", "C1 tier-1 (genuine clean, staged)"),
    PairSpec((243, 65), (243, 66), "c1_matched_staged", "C1 tier-1 (genuine clean, staged)"),
    PairSpec((250, 10), (250, 11), "c1_matched_staged", "C1 tier-1 (genuine clean, staged)"),
    PairSpec(
        (216, 106),
        (216, 107),
        "c4_tier2_coset_confounded",
        "C4 tier-2 (FS-identical, coset-confounded)",
    ),
    PairSpec((64, 60), (64, 65), "fs_flip_control", "C4 tier-3 (FS_FLIP, same CT)"),
    PairSpec((104, 4), (104, 6), "fs_flip_control", "C4 tier-3 (FS_FLIP, same CT)"),
    PairSpec((32, 18), (32, 20), "fs_flip_control", "C2 D32/Q32 (FS_FLIP, same CT)"),
    PairSpec(
        (48, 28), (48, 29), "not_ct_equal_control", "C5 (NOT CT-equivalent -- different tables)"
    ),
    PairSpec(
        (192, 10), (192, 24), "not_ct_equal_control", "E8 (NOT CT-equivalent -- different tables)"
    ),
]

#: The six c1_matched pairs are the ones the used-set-audit robustness pass
#: (noise floor + bootstrap-seed check) is scoped to.
C1_MATCHED_PAIR_LABELS = [
    f"({a[0]},{a[1]})/({b[0]},{b[1]})" for a, b, cat, _ in PAIRS if cat == "c1_matched"
]


def glob_cell_files(instr_dir: Path, gid: GroupId) -> dict[str, Path]:
    """Return {width_suffix: path} for a group id under an instrument's
    results directory, e.g. {'w128': ..., 'w256': ...}."""
    o, i = gid
    out: dict[str, Path] = {}
    if not instr_dir.exists():
        return out
    for f in instr_dir.iterdir():
        parts = f.stem.split("_")
        try:
            idx = parts.index(str(o))
        except ValueError:
            continue
        if idx + 1 < len(parts) and parts[idx + 1].split("w")[0] == str(i):
            width_key = "_".join(parts[idx + 2 :])
            out[width_key] = f
    return out


def load_runs(path: Path) -> list[dict[str, Any]]:
    runs = json.loads(path.read_text()).get("runs", [])
    assert isinstance(runs, list)
    return runs


# ---- per-instrument headline extractors -------------------------------------
def extract_occupancy(runs: list[dict[str, Any]]) -> dict[int, float]:
    """occupancy.left.full.tv_to_null -- TV distance to the group's own
    analytic null, left argument. Matches the 'left_full' statistic already
    reported in results/occupancy/c1_occupancy_contrast.json."""
    out: dict[int, float] = {}
    for r in runs:
        if r.get("status") != "measured":
            continue
        occ = r.get("occupancy")
        if not occ:
            continue
        v = occ.get("left", {}).get("full", {}).get("tv_to_null")
        if v is not None:
            out[r["seed"]] = v
    return out


def extract_recruited_dimension(runs: list[dict[str, Any]]) -> dict[int, float]:
    """recruited_dimension.discrimination.ratios.minimal_faithful_real --
    recruited dimension over the minimal-faithful-real anchor."""
    out: dict[int, float] = {}
    for r in runs:
        if r.get("status") != "measured":
            continue
        rd = r.get("recruited_dimension")
        if not rd:
            continue
        v = rd.get("discrimination", {}).get("ratios", {}).get("minimal_faithful_real")
        if v is not None:
            out[r["seed"]] = v
    return out


def extract_used_set_audit_sufficiency(runs: list[dict[str, Any]]) -> dict[int, float]:
    """used_set_audit.completeness.accuracy_retention_fraction.zero -- held-
    out accuracy kept when the model is restricted to its occupied blocks."""
    out: dict[int, float] = {}
    for r in runs:
        if r.get("status") != "measured":
            continue
        usa = r.get("used_set_audit")
        if not usa:
            continue
        v = usa.get("completeness", {}).get("accuracy_retention_fraction", {}).get("zero")
        if v is not None:
            out[r["seed"]] = v
    return out


def extract_used_set_audit_necessity(runs: list[dict[str, Any]]) -> dict[int, float]:
    """used_set_audit.n_necessary_blocks / used_set_audit.n_used_blocks --
    fraction of the occupied blocks individually necessary beyond the random
    control. Seeds with n_used_blocks == 0 (the whole used set is the trivial
    block) are skipped, not divided by zero."""
    out: dict[int, float] = {}
    for r in runs:
        if r.get("status") != "measured":
            continue
        usa = r.get("used_set_audit")
        if not usa:
            continue
        n_nec = usa.get("n_necessary_blocks")
        n_used = usa.get("n_used_blocks")
        if n_nec is None or not n_used:
            continue
        out[r["seed"]] = n_nec / n_used
    return out


def extract_gcr_matmul(runs: list[dict[str, Any]]) -> dict[int, float]:
    """gcr_matmul.fits[argmax occupancy].mp_fve_heldout -- matrix-product FVE
    at the seed's highest-occupancy irrep block. Disclosed convention: the
    instrument fits every block of the relevant irrep degree, and there is no
    single designated "the" block in the record, so the highest-occupancy
    block is used as the per-seed headline."""
    out: dict[int, float] = {}
    for r in runs:
        if r.get("status") != "measured":
            continue
        gm = r.get("gcr_matmul")
        if not gm or not gm.get("fits"):
            continue
        best = max(gm["fits"], key=lambda f: f["occupancy"])
        out[r["seed"]] = best["mp_fve_heldout"]
    return out


def extract_readout_characterisation(runs: list[dict[str, Any]]) -> dict[int, float]:
    """readout_characterisation.held_out_fve_gain_full_minus_character --
    full irrep-matrix entries vs. the scalar character of the same irreps,
    held-out FVE gain."""
    out: dict[int, float] = {}
    for r in runs:
        if r.get("status") != "measured":
            continue
        rc = r.get("readout_characterisation")
        if not rc:
            continue
        v = rc.get("held_out_fve_gain_full_minus_character")
        if v is not None:
            out[r["seed"]] = v
    return out


def extract_gcr_readout(runs: list[dict[str, Any]]) -> dict[int, float]:
    """gcr_readout.primary.nested_comparison.full_vs_fourier_held_out_fve_gain
    -- full irrep matrices vs. the abelianisation-only (degree-1 irreps)
    readout, held-out FVE gain."""
    out: dict[int, float] = {}
    for r in runs:
        if r.get("status") != "measured":
            continue
        gr = r.get("gcr_readout")
        if not gr:
            continue
        v = (
            gr.get("primary", {})
            .get("nested_comparison", {})
            .get("full_vs_fourier_held_out_fve_gain")
        )
        if v is not None:
            out[r["seed"]] = v
    return out


def extract_coset_quotient(runs: list[dict[str, Any]]) -> dict[int, float]:
    """coset_quotient_route.quotients[decisive_subgroup_index].necessity
    .coset_accuracy_drop_over_random -- at the seed's own decisive quotient
    (the one maximising sufficiency.coset_accuracy_over_random, per
    instruments/coset_quotient.py), how much ablating it costs coset
    accuracy over a matched random subspace."""
    out: dict[int, float] = {}
    for r in runs:
        if r.get("status") != "measured":
            continue
        cq = r.get("coset_quotient_route")
        if not cq or cq.get("defined") is not True:
            continue
        dsi = cq.get("decisive_subgroup_index")
        if dsi is None:
            continue
        match = next((q for q in cq.get("quotients", []) if q["subgroup_index"] == dsi), None)
        if match is None:
            continue
        v = match.get("necessity", {}).get("coset_accuracy_drop_over_random")
        if v is not None:
            out[r["seed"]] = v
    return out


#: instrument key -> (results/ subdirectory name, extractor, headline field description)
INSTRUMENTS: dict[str, tuple[str, Any, str]] = {
    "occupancy": ("occupancy", extract_occupancy, "occupancy.left.full.tv_to_null"),
    "recruited_dimension": (
        "recruited_dimension",
        extract_recruited_dimension,
        "recruited_dimension.discrimination.ratios.minimal_faithful_real",
    ),
    "used_set_audit_sufficiency": (
        "used_set_audit",
        extract_used_set_audit_sufficiency,
        "used_set_audit.completeness.accuracy_retention_fraction.zero",
    ),
    "used_set_audit_necessity": (
        "used_set_audit",
        extract_used_set_audit_necessity,
        "used_set_audit.n_necessary_blocks / used_set_audit.n_used_blocks",
    ),
    "gcr_matmul": (
        "gcr_matmul",
        extract_gcr_matmul,
        "gcr_matmul.fits[argmax occupancy].mp_fve_heldout",
    ),
    "readout_characterisation": (
        "readout_characterisation",
        extract_readout_characterisation,
        "readout_characterisation.held_out_fve_gain_full_minus_character",
    ),
    "gcr_readout": (
        "gcr_readout",
        extract_gcr_readout,
        "gcr_readout.primary.nested_comparison.full_vs_fourier_held_out_fve_gain",
    ),
    "coset_quotient": (
        "coset_quotient",
        extract_coset_quotient,
        "coset_quotient_route.quotients[decisive_subgroup_index]"
        ".necessity.coset_accuracy_drop_over_random",
    ),
}


# ---- paired sign-flip permutation test ---------------------------------------
def sign_flip_permutation_test(
    diffs: list[float], *, seed: int = 0, n_resamples: int = 20_000, exact_max_n: int = 16
) -> dict[str, Any]:
    """Two-sided sign-flip permutation test on the mean of paired differences.

    Swapping which member is "A" vs "B" for a seed flips that seed's diff
    sign, so this is exactly the paired-relabelling null. Exact enumeration of
    all 2**n sign patterns for n <= exact_max_n; otherwise Monte Carlo with a
    private seeded RNG (mirrors stats.permutation_test's own convention:
    p = (as_extreme + 1) / (n_resamples + 1), so p is never exactly zero)."""
    n = len(diffs)
    observed = abs(statistics.fmean(diffs))
    if n <= exact_max_n:
        as_extreme = 0
        total = 1 << n
        for bits in range(total):
            s = 0.0
            for i, d in enumerate(diffs):
                s += d if (bits >> i) & 1 else -d
            if abs(s / n) >= observed - 1e-12:
                as_extreme += 1
        return {
            "p_two_sided": as_extreme / total,
            "n_permutations": total,
            "exact": True,
            "observed_abs_mean": observed,
        }
    rng = random.Random(seed)
    as_extreme = 0
    for _ in range(n_resamples):
        s = sum(d if rng.random() < 0.5 else -d for d in diffs) / n
        if abs(s) >= observed - 1e-12:
            as_extreme += 1
    return {
        "p_two_sided": (as_extreme + 1) / (n_resamples + 1),
        "n_permutations": n_resamples,
        "exact": False,
        "observed_abs_mean": observed,
    }


def benjamini_hochberg(pvals: list[float], q: float) -> list[bool]:
    """Standard BH step-up procedure; returns a reject list aligned to input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    reject = [False] * m
    max_k = -1
    for rank, i in enumerate(order, start=1):
        if pvals[i] <= (rank / m) * q:
            max_k = rank
    if max_k >= 0:
        for rank, i in enumerate(order, start=1):
            if rank <= max_k:
                reject[i] = True
    return reject


def binomial_upper_tail_p(n: int, k: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p), computed exactly via math.comb."""
    return sum(math.comb(n, i) * (p**i) * ((1 - p) ** (n - i)) for i in range(k, n + 1))


def pearson(a: list[float], b: list[float]) -> float:
    n = len(a)
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b, strict=True)) / (n - 1)
    sa = statistics.stdev(a) if n > 1 else 0.0
    sb = statistics.stdev(b) if n > 1 else 0.0
    if sa == 0.0 or sb == 0.0:
        return float("nan")
    return cov / (sa * sb)


def noise_floor_summary(
    a_vals: list[float], b_vals: list[float], diffs: list[float]
) -> dict[str, float]:
    n = len(diffs)
    mean_diff = statistics.fmean(diffs)
    sd_a = statistics.stdev(a_vals) if n > 1 else 0.0
    sd_b = statistics.stdev(b_vals) if n > 1 else 0.0
    sd_diff = statistics.stdev(diffs) if n > 1 else 0.0
    naive_unpaired_sd = math.sqrt(sd_a**2 + sd_b**2)
    avg_seed_sd = (sd_a + sd_b) / 2
    return {
        "n": n,
        "mean_diff": mean_diff,
        "sd_a_within_member": sd_a,
        "sd_b_within_member": sd_b,
        "sd_paired_diff": sd_diff,
        "ab_correlation": pearson(a_vals, b_vals),
        "naive_unpaired_sd_if_independent": naive_unpaired_sd,
        "pairing_variance_reduction_fraction": (
            1 - sd_diff / naive_unpaired_sd if naive_unpaired_sd > 0 else float("nan")
        ),
        "paired_cohens_dz": mean_diff / sd_diff if sd_diff > 0 else float("nan"),
        "abs_mean_diff_over_avg_seed_sd": (
            abs(mean_diff) / avg_seed_sd if avg_seed_sd > 0 else float("nan")
        ),
    }


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---- main computation ---------------------------------------------------------
def compute_contrasts(
    results_dir: Path, *, bootstrap_seed: int, permutation_seed: int
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Returns (contrasts, input_files_sha256). ``contrasts`` has one entry per
    (instrument, pair, width) with committed data for both members, whether or
    not there were enough paired seeds to compute a statistic."""
    contrasts: list[dict[str, Any]] = []
    input_sha: dict[str, str] = {}

    for instr_key, (dirname, extractor, headline) in INSTRUMENTS.items():
        instr_dir = results_dir / dirname
        for pair in PAIRS:
            fa = glob_cell_files(instr_dir, pair.a)
            fb = glob_cell_files(instr_dir, pair.b)
            common_widths = sorted(set(fa) & set(fb))
            pair_name = f"({pair.a[0]},{pair.a[1]})/({pair.b[0]},{pair.b[1]})"
            if not common_widths:
                contrasts.append(
                    {
                        "instrument": instr_key,
                        "headline_field": headline,
                        "pair": pair_name,
                        "category": pair.category,
                        "label": pair.label,
                        "status": "missing",
                    }
                )
                continue
            for w in common_widths:
                for f in (fa[w], fb[w]):
                    rel = str(f.relative_to(REPO))
                    if rel not in input_sha:
                        input_sha[rel] = file_sha256(f)
                va = extractor(load_runs(fa[w]))
                vb = extractor(load_runs(fb[w]))
                paired_seeds = sorted(set(va) & set(vb))
                n_paired = len(paired_seeds)
                entry: dict[str, Any] = {
                    "instrument": instr_key,
                    "headline_field": headline,
                    "pair": f"{pair_name} {w}",
                    "category": pair.category,
                    "label": pair.label,
                    "status": "computed" if n_paired >= 2 else "insufficient_paired_seeds",
                    "n_a_measured": len(va),
                    "n_b_measured": len(vb),
                    "n_paired": n_paired,
                }
                if n_paired >= 2:
                    a_vals = [va[s] for s in paired_seeds]
                    b_vals = [vb[s] for s in paired_seeds]
                    diffs = [x - y for x, y in zip(a_vals, b_vals, strict=True)]
                    mean = statistics.fmean(diffs)
                    lo, hi = stats.bootstrap_ci(diffs, seed=bootstrap_seed)
                    perm = sign_flip_permutation_test(diffs, seed=permutation_seed)
                    entry["diff_a_minus_b"] = {
                        "mean": mean,
                        "bootstrap_ci_95": [lo, hi],
                        "clears_zero": (lo > 0.0) or (hi < 0.0),
                    }
                    entry["permutation_test"] = perm
                    entry["paired_seeds"] = paired_seeds
                    entry["a_values"] = a_vals
                    entry["b_values"] = b_vals
                    entry["per_seed_diffs"] = diffs
                contrasts.append(entry)
    return contrasts, input_sha


def multiple_looks_summary(
    contrasts: list[dict[str, Any]], *, q_values: tuple[float, ...]
) -> dict[str, Any]:
    computed = [c for c in contrasts if c["status"] == "computed"]
    n_total = len(computed)
    n_clears = sum(1 for c in computed if c["diff_a_minus_b"]["clears_zero"])
    pvals = [c["permutation_test"]["p_two_sided"] for c in computed]

    bh: dict[str, list[bool]] = {}
    for q in q_values:
        bh[f"q{q:g}"] = benjamini_hochberg(pvals, q)
    for i, c in enumerate(computed):
        for q in q_values:
            c[f"bh_reject_q{q:g}"] = bh[f"q{q:g}"][i]

    falsifier_relevant = [
        c for c in computed if c["category"] in ("c1_matched", "c4_tier2_coset_confounded")
    ]
    n_fr = len(falsifier_relevant)
    n_fr_clears = sum(1 for c in falsifier_relevant if c["diff_a_minus_b"]["clears_zero"])

    return {
        "note": (
            "p_permutation is the two-sided paired sign-flip permutation p-value on "
            "the same per-seed diffs used for the bootstrap CI. bh_reject_q* is the "
            "Benjamini-Hochberg step-up decision over ALL computed contrasts' "
            "permutation p-values in one pass (every instrument, every pair, every "
            "width -- the full battery of independent looks actually taken)."
        ),
        "n_total_looks": n_total,
        "n_clears_zero_by_ci": n_clears,
        "expected_false_positives_at_5pct_per_look": 0.05 * n_total,
        "p_binomial_upper_tail_ge_observed_given_p05": binomial_upper_tail_p(
            n_total, n_clears, 0.05
        ),
        "n_bh_reject_q0.05": sum(bh["q0.05"]),
        "n_bh_reject_q0.1": sum(bh["q0.1"]),
        "falsifier_relevant_subset": {
            "definition": "category in {c1_matched, c4_tier2_coset_confounded}",
            "n_looks": n_fr,
            "n_clears_zero_by_ci": n_fr_clears,
            "p_binomial_upper_tail_ge_observed_given_p05": binomial_upper_tail_p(
                n_fr, n_fr_clears, 0.05
            ),
        },
    }


def used_set_stress_test(
    contrasts: list[dict[str, Any]], *, bootstrap_seeds: tuple[int, ...]
) -> dict[str, Any]:
    """Sufficiency-vs-necessity deep dive, noise-floor comparison, and
    bootstrap-seed robustness for the six c1_matched pairs' used-set-audit
    contrasts -- the check the coordinator asked to stress-test."""
    by_key = {(c["instrument"], c["pair"]): c for c in contrasts if c["status"] == "computed"}
    deep: list[dict[str, Any]] = []
    for pair_label in C1_MATCHED_PAIR_LABELS:
        # widths vary per pair; find whichever computed entry matches this pair's label prefix
        suff = next(
            (
                c
                for (instr, p), c in by_key.items()
                if instr == "used_set_audit_sufficiency" and p.startswith(pair_label)
            ),
            None,
        )
        nec = next(
            (
                c
                for (instr, p), c in by_key.items()
                if instr == "used_set_audit_necessity" and p.startswith(pair_label)
            ),
            None,
        )
        entry: dict[str, Any] = {"pair": pair_label}
        for label, c in (("sufficiency", suff), ("necessity", nec)):
            if c is None:
                entry[label] = {"status": "missing_or_insufficient"}
                continue
            diffs = c["per_seed_diffs"]
            noise = noise_floor_summary(c["a_values"], c["b_values"], diffs)
            robustness = {}
            for bseed in bootstrap_seeds:
                lo, hi = stats.bootstrap_ci(diffs, seed=bseed)
                robustness[str(bseed)] = {"ci": [lo, hi], "clears_zero": (lo > 0.0) or (hi < 0.0)}
            entry[label] = {
                "resolved_pair_width": c["pair"],
                "n_paired": c["n_paired"],
                "mean": c["diff_a_minus_b"]["mean"],
                "bootstrap_ci_95_seed0": c["diff_a_minus_b"]["bootstrap_ci_95"],
                "clears_zero_ci": c["diff_a_minus_b"]["clears_zero"],
                "permutation_p": c["permutation_test"]["p_two_sided"],
                "noise_floor": noise,
                "bootstrap_seed_robustness": robustness,
            }
        deep.append(entry)
    return {"six_target_pairs": deep}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--permutation-seed", type=int, default=0)
    args = parser.parse_args()

    contrasts, input_sha = compute_contrasts(
        args.results_dir, bootstrap_seed=args.bootstrap_seed, permutation_seed=args.permutation_seed
    )
    looks = multiple_looks_summary(contrasts, q_values=(0.05, 0.1))
    stress = used_set_stress_test(contrasts, bootstrap_seeds=(0, 1, 2, 42, 123))

    used_set_bh_survivors = [
        c
        for c in contrasts
        if c["status"] == "computed"
        and c["instrument"] == "used_set_audit_sufficiency"
        and c.get("bh_reject_q0.05")
    ]
    # The falsification-relevant subset of the above: c1_matched pairs only,
    # excluding not_ct_equal_control survivors (a nonzero contrast there is
    # expected by design -- the pair does not share a character table -- so
    # it carries no C1 falsification weight even though it also survives BH).
    used_set_bh_survivors_c1_matched = [
        c for c in used_set_bh_survivors if c["category"] == "c1_matched"
    ]

    out = {
        "instrument": "within-pair-contrasts",
        "provenance": {
            "producer": "scripts/derive_within_pair_contrasts.py",
            "producer_git_commit": get_git_commit(),
            "producer_git_dirty": get_git_dirty(),
            "bootstrap_seed": args.bootstrap_seed,
            "permutation_seed": args.permutation_seed,
            "input_files_sha256": dict(sorted(input_sha.items())),
        },
        "headline_fields": {k: v[2] for k, v in INSTRUMENTS.items()},
        "pair_categories": {
            cat: [f"({a[0]},{a[1]})/({b[0]},{b[1]})" for a, b, c, _ in PAIRS if c == cat]
            for cat in dict.fromkeys(p.category for p in PAIRS)
        },
        "contrasts": contrasts,
        "multiple_looks": looks,
        "used_set_audit_sufficiency_bh_survivors_q0.05": used_set_bh_survivors,
        "used_set_audit_sufficiency_bh_survivors_c1_matched_only": used_set_bh_survivors_c1_matched,
        "used_set_stress_test": stress,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))

    n_computed = sum(1 for c in contrasts if c["status"] == "computed")
    print(f"wrote {args.output}")
    print(f"{len(contrasts)} (instrument, pair, width) entries; {n_computed} computed")
    print(
        f"multiple looks: {looks['n_clears_zero_by_ci']}/{looks['n_total_looks']} clear zero by CI; "
        f"BH q=0.05 rejects {looks['n_bh_reject_q0.05']}"
    )
    print(
        f"used_set_audit_sufficiency BH q=0.05 survivors: {[c['pair'] for c in used_set_bh_survivors]}"
    )
    print(
        f"...of which c1_matched (falsifier-relevant): "
        f"{[c['pair'] for c in used_set_bh_survivors_c1_matched]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
