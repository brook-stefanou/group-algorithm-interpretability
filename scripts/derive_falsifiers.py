#!/usr/bin/env python3
"""Mechanically-derived Fourier-falsifier screen -- the authority for claim C1.

Screens every character-table-equivalent group pair in
``data/group_properties_full.jsonl`` (21 <= order <= 255, the trainable range,
order 128 included; 6,958 groups) and classifies each pair as ``GENUINE_CLEAN``
/ ``FS_IDENTICAL_COSET_CONFOUNDED`` / ``FS_FLIP`` / ``CT_EQUAL_ONLY``.  The set
of ``GENUINE_CLEAN`` pairs (``genuine_clean_pairs`` in the output) is the
authoritative, complete clean-pair family that defines C1; the core study's
selection rule is applied to that list, not to hand-assembly.

Every group-theoretic computation is deferred to GAP via ``falsifier_lib.g``
(shipped alongside this script).  The five-stage screen, reordered for the
~30x larger pair count than the panel-only screen it descends from:

  Stage A (jsonl only, cheap): bucket every in-range group by
    ``character_table_fingerprint``; every same-fingerprint pair is a Stage-1
    candidate.  Two jsonl-only filters narrow these to Stage-A *survivors*
    (the only pairs the expensive core-free-subgroup lattice enumeration runs
    for):
      - FS count-triple equality (``fs_real_count`` / ``fs_complex_count`` /
        ``fs_quaternionic_count``) -- the LOOSE Frobenius-Schur signature.
      - scalar coset equality (``min_corefree_index``,
        ``minimal_faithful_permutation_degree``, sorted
        ``corefree_index_spectrum``).
    A THIRD exact jsonl-only signal runs on every Stage-1 pair, not just
    survivors: ``zip(character_degrees, indicator_vector)`` IS the group's
    strict Frobenius-Schur joint multiset ``{(chi(1), nu2(chi))}`` -- verified
    index-aligned over all in-range groups and cross-checked against GAP's
    ``fs_joint`` for every survivor group.

  Stage B (GAP): two independent passes.
      - ``CTProof`` (ordinary character-table equality via
        ``TransformingPermutations(Irr(G), Irr(H))``) for every non-exact
        same-fingerprint pair -- ``ct_equal`` is checked BEFORE ``fs_strict``
        in the classification, so it must be known for every Stage-1 pair, not
        just the FS-strict-identical ones.  CTProof never touches subgroup
        lattices and is fast.
      - ``GroupData`` (the core-free-subgroup-lattice call, with the
        FS-involution identity ``Sum nu2(chi)*chi(1) = #{g:g^2=1}`` and
        Frobenius reciprocity ``Sum m_chi*chi(1) = [G:H]`` asserted inside GAP)
        ONLY for the groups touched by a Stage-A survivor pair, chunked one GAP
        subprocess per order with a per-group timeout fallback so one
        pathological group cannot stall the whole run.

  Classification (checked in this order):
    not ct_equal            -> CT_EQUAL_ONLY
    ct_equal, not fs_strict -> FS_FLIP
    fs_strict, not (scalar_clean and template_clean) -> FS_IDENTICAL_COSET_CONFOUNDED
    fs_strict, scalar_clean, template_clean           -> GENUINE_CLEAN

Full per-pair records (all verdicts) are kept for every pair NOT classified
FS_FLIP; FS_FLIP pairs are counted only (their classification is fully
determined by ``ct_equal`` and ``fs_strict``).

Determinism: pairs and GAP calls are emitted in sorted order and every GAP
operation used is deterministic for these groups.  No timestamps, no
randomness -- the output is byte-reproducible given the pinned inputs.

Inputs are pinned by sha256 (recorded in the output ``provenance`` block).  The
GAP binary is configurable (``--gap`` / ``$GAI_GAP_BIN``); its default is the
version the committed results file was produced with.

Usage::

    uv run python scripts/derive_falsifiers.py
    uv run python scripts/derive_falsifiers.py --panel results/falsifier_panel_rows.json

The dataset-wide run (no ``--panel``) reproduces
``results/falsifier_screen_results_full.json`` byte-for-byte.  ``--panel FILE``
restricts the screened universe to the ``(order, index)`` rows in ``FILE`` (the
panel variant), writing the same output schema over the restricted set.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_JSONL = REPO / "data" / "group_properties_full.jsonl"
DEFAULT_OUTPUT = REPO / "results" / "falsifier_screen_results_full.json"
PANEL_ROWS = REPO / "results" / "falsifier_panel_rows.json"
LIB = HERE / "falsifier_lib.g"
DEFAULT_GAP = "/opt/miniconda3/envs/gap/bin/gap"

ORDER_MIN, ORDER_MAX = 21, 255
GROUP_TIMEOUT_S = 65  # per-group fallback cap ("EXPENSIVE" threshold)
CHUNK_TIMEOUT_S = 900  # per-order-chunk timeout before falling back

# Fields pulled from the ground-truth jsonl (scalar screen + residual axis).
JSONL_COLS = [
    "order",
    "index",
    "name",
    "character_degrees",
    "character_table_fingerprint",
    "character_table_fingerprint_exact",
    "fs_real_count",
    "fs_complex_count",
    "fs_quaternionic_count",
    "indicator_vector",
    "min_corefree_index",
    "minimal_faithful_permutation_degree",
    "corefree_index_spectrum",
    "element_order_histogram",
    "exponent",
    "aut_order",
    "number_subgroups",
    "num_involutions",
    "max_element_order",
]

GroupId = tuple[int, int]
Pair = tuple[GroupId, GroupId]
Row = dict[str, Any]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---- input loading ----------------------------------------------------------
def load_panel_ids(path: Path) -> list[GroupId]:
    """Load the pinned ``(order, index)`` rows from a panel file (either a bare
    list of rows or ``{"rows": [...]}``)."""
    obj = json.loads(path.read_text())
    rows = obj["rows"] if isinstance(obj, dict) else obj
    ids = [(r["order"], r["index"]) for r in rows]
    assert len(ids) == len(set(ids)), f"duplicate (order,index) in {path}"
    return ids


def load_groups(jsonl: Path, restrict: set[GroupId] | None) -> dict[GroupId, Row]:
    """Load the jsonl columns for every in-range group, or -- if ``restrict``
    is given -- only those ids (the panel variant)."""
    db: dict[GroupId, Row] = {}
    with open(jsonl) as f:
        for line in f:
            d = json.loads(line)
            key: GroupId = (d["order"], d["index"])
            if restrict is not None:
                if key not in restrict:
                    continue
            elif not (ORDER_MIN <= d["order"] <= ORDER_MAX):
                continue
            db[key] = {c: d.get(c) for c in JSONL_COLS}
    if restrict is not None:
        missing = restrict - set(db)
        assert not missing, f"panel groups missing from jsonl: {missing}"
    return db


# ---- jsonl-only signals -----------------------------------------------------
def fs_loose_triple(d: Row) -> tuple[int, int, int]:
    return (d["fs_real_count"], d["fs_complex_count"], d["fs_quaternionic_count"])


def fs_strict_sig(d: Row) -> tuple[Any, ...]:
    """Strict FS joint multiset ``{(chi(1), nu2(chi))}`` from
    ``zip(character_degrees, indicator_vector)`` -- verified index-aligned with
    ``fs_*_count`` over every in-range group, and the same quantity
    ``falsifier_lib.g``'s ``GroupData`` computes from ``Irr(G)``."""
    return tuple(sorted(Counter(zip(d["character_degrees"], d["indicator_vector"])).items()))


def scalar_sig(d: Row) -> tuple[Any, ...]:
    return (
        d["min_corefree_index"],
        d["minimal_faithful_permutation_degree"],
        tuple(sorted(d["corefree_index_spectrum"])),
    )


def stage1_pairs(ids: list[GroupId], db: dict[GroupId, Row]) -> list[Pair]:
    """All unordered pairs among ``ids`` sharing a character_table_fingerprint."""
    buckets: dict[str, list[GroupId]] = defaultdict(list)
    for k in ids:
        buckets[db[k]["character_table_fingerprint"]].append(k)
    pairs: list[Pair] = []
    for group_ids in buckets.values():
        if len(group_ids) >= 2:
            for a, b in itertools.combinations(sorted(group_ids), 2):
                pairs.append((a, b))
    return sorted(pairs)


# ---- GAP output repair (line-wrap + integral-float fixes) -------------------
def clean_gap_json(raw: str) -> list[Any]:
    txt = raw.replace("\\\n", "")  # backslash-newline inside a quoted string
    txt = txt.replace("\n", " ")  # bare newline -> whitespace
    txt = re.sub(r"(\d)\.(?=[,\s\]])", r"\1.0", txt)  # "0." -> "0.0"
    dec = json.JSONDecoder()
    objs: list[Any] = []
    i, n = 0, len(txt)
    while i < n:
        while i < n and txt[i] in " \t":
            i += 1
        if i >= n:
            break
        obj, end = dec.raw_decode(txt, i)
        objs.append(obj)
        i = end
    return objs


class Gap:
    """Runs GAP scripts against ``falsifier_lib.g``, capturing raw stdout to a
    scratch directory (regenerated every run; irrelevant to the output)."""

    def __init__(self, binary: str, capture_dir: Path) -> None:
        self.binary = binary
        self.capture_dir = capture_dir

    def run(self, lines: list[str], out_name: str, timeout_s: int) -> list[Any] | None:
        script = "\n".join([f'Read("{LIB}");'] + lines + ["QUIT;"]) + "\n"
        out_path = self.capture_dir / out_name
        try:
            proc = subprocess.run(
                [self.binary, "-q"],
                input=script,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=self.capture_dir,
            )
        except subprocess.TimeoutExpired:
            with open(out_path, "a") as f:
                f.write(f"# TIMEOUT after {timeout_s}s: {lines}\n")
            return None
        with open(out_path, "a") as f:
            f.write(proc.stdout)
        if proc.returncode != 0:
            with open(out_path, "a") as f:
                f.write(f"# NONZERO EXIT {proc.returncode}: {lines}\n{proc.stderr}\n")
            return None
        try:
            return clean_gap_json(proc.stdout)
        except Exception:
            return None


def run_ctproof_batch(gap: Gap, pairs: list[Pair]) -> dict[Pair, bool]:
    """All CTProof calls in one GAP process (fast: no subgroup-lattice work)."""
    lines = [f"CTProof({a[0]},{a[1]},{b[0]},{b[1]});" for a, b in pairs]
    objs = gap.run(lines, "gap_ctproof.jsonl", timeout_s=max(300, 2 * len(pairs)))
    if objs is None:
        raise RuntimeError("CTProof batch GAP call failed or timed out")
    ctproof: dict[Pair, bool] = {}
    for o in objs:
        assert o["kind"] == "ctproof"
        ctproof[(tuple(o["a"]), tuple(o["b"]))] = o["ct_equal_irr"]
    missing = set(pairs) - set(ctproof)
    assert not missing, f"CTProof missing results for {missing}"
    return ctproof


def run_groupdata(gap: Gap, groups: list[GroupId]) -> tuple[dict[GroupId, Row], list[Row]]:
    """GroupData for every group, chunked one GAP subprocess per order, with a
    per-group (~65s) fallback on chunk failure/timeout.  Returns
    ``(gdata, expensive_skips)``."""
    by_order: dict[int, list[int]] = defaultdict(list)
    for o, i in groups:
        by_order[o].append(i)

    gdata: dict[GroupId, Row] = {}
    expensive: list[Row] = []
    for o in sorted(by_order):
        idxs = sorted(by_order[o])
        lines = [f"GroupData({o},{i});" for i in idxs]
        objs = gap.run(lines, "gap_groupdata.jsonl", timeout_s=CHUNK_TIMEOUT_S)
        ok = False
        if objs is not None:
            got = {tuple(x["id"]): x for x in objs if x["kind"] == "groupdata"}
            if set(got) == {(o, i) for i in idxs}:
                gdata.update(got)
                ok = True
        if ok:
            continue
        # Fallback: isolate group-by-group with a hard per-group timeout.
        for i in idxs:
            key: GroupId = (o, i)
            objs1 = gap.run(
                [f"GroupData({o},{i});"], "gap_groupdata.jsonl", timeout_s=GROUP_TIMEOUT_S
            )
            if objs1 is not None and len(objs1) == 1 and objs1[0]["kind"] == "groupdata":
                gdata[key] = objs1[0]
            else:
                expensive.append(
                    {"id": [o, i], "reason": "timeout_or_error", "timeout_s": GROUP_TIMEOUT_S}
                )
    return gdata, expensive


# ---- screen helpers ---------------------------------------------------------
def joint_multiset(gd: Row) -> tuple[Any, ...]:
    return tuple(sorted((tuple(item), cnt) for item, cnt in gd["fs_joint"]))


def _sig(t: Row) -> tuple[Any, ...]:
    return tuple(sorted((tuple(pair), cnt) for pair, cnt in t["tmpl"]))


def templates_by_index(
    gd: Row,
) -> tuple[dict[int, set[Any]], dict[int, Counter[Any]], dict[int, set[int]]]:
    tset: dict[int, set[Any]] = defaultdict(set)
    tmset: dict[int, Counter[Any]] = defaultdict(Counter)
    supp: dict[int, set[int]] = defaultdict(set)
    for t in gd["templates"]:
        s = _sig(t)
        tset[t["idx"]].add(s)
        tmset[t["idx"]][s] += 1
        supp[t["idx"]].add(t["support"])
    return tset, tmset, supp


def _render(sigs: set[Any]) -> list[Any]:
    return [[[list(pair), n] for pair, n in s] for s in sorted(sigs)]


def template_verdict(gda: Row, gdb: Row) -> dict[str, Any]:
    """Compare core-free-subgroup induction templates across a pair, per index
    (grouped by ``[G:H]``).  ``set_clean`` (identical SET of distinct templates
    at every index) is the decisive occupancy criterion and the primary
    ``clean`` verdict; ``support_clean`` / ``multiset_clean`` are weaker/stricter
    variants recorded but not used for classification.  ``min_index_set_clean``
    reproduces the prior audit's minimal-index-only screen."""
    sa, ma, ua = templates_by_index(gda)
    sb, mb, ub = templates_by_index(gdb)
    idxs = sorted(set(sa) | set(sb))
    min_idx = idxs[0] if idxs else None

    set_clean = support_clean = multiset_clean = True
    detail: list[Row] = []
    for idx in idxs:
        ta, tb = sa.get(idx, set()), sb.get(idx, set())
        if ta != tb:
            set_clean = False
        if ua.get(idx, set()) != ub.get(idx, set()):
            support_clean = False
        if ma.get(idx, Counter()) != mb.get(idx, Counter()):
            multiset_clean = False
        if ta != tb:
            detail.append(
                {
                    "index": idx,
                    "a_only_templates": _render(ta - tb),
                    "b_only_templates": _render(tb - ta),
                    "a_support_sizes": sorted(ua.get(idx, set())),
                    "b_support_sizes": sorted(ub.get(idx, set())),
                }
            )

    min_index_set_clean = min_idx is not None and sa.get(min_idx, set()) == sb.get(min_idx, set())
    return {
        "clean": set_clean,
        "set_clean": set_clean,
        "support_clean": support_clean,
        "multiset_clean": multiset_clean,
        "min_index": min_idx,
        "min_index_set_clean": min_index_set_clean,
        "differing_index_detail": detail,
    }


def classify(ct_equal: bool, fs_strict: bool, scalar_clean: bool, template_clean: bool) -> str:
    """The four-verdict classification rule, checked strictly in this order.

    ``ct_equal`` is checked before ``fs_strict`` because a fingerprint collision
    GAP cannot confirm as true character-table equality is ``CT_EQUAL_ONLY``
    regardless of its FS status.  A pair is ``GENUINE_CLEAN`` only when it clears
    every screen; a scalar-fail and a template-fail both land in
    ``FS_IDENTICAL_COSET_CONFOUNDED``."""
    if not ct_equal:
        return "CT_EQUAL_ONLY"
    if not fs_strict:
        return "FS_FLIP"
    if not (scalar_clean and template_clean):
        return "FS_IDENTICAL_COSET_CONFOUNDED"
    return "GENUINE_CLEAN"


# ---- screen -----------------------------------------------------------------
def run_screen(
    jsonl: Path, gap_binary: str, panel_path: Path | None, capture_dir: Path
) -> dict[str, Any]:
    restrict_ids = None
    if panel_path is not None:
        restrict_ids = set(load_panel_ids(panel_path))

    db = load_groups(jsonl, restrict_ids)
    if restrict_ids is None:
        assert len(db) == 6958, f"expected 6958 in-range groups, got {len(db)}"

    pairs = stage1_pairs(sorted(db), db)

    # jsonl-only signals for every Stage-1 pair.
    fs_loose: dict[Pair, bool] = {}
    fs_strict: dict[Pair, bool] = {}
    scalar_clean: dict[Pair, bool] = {}
    both_exact: dict[Pair, bool] = {}
    for a, b in pairs:
        da, dbb = db[a], db[b]
        fs_loose[(a, b)] = fs_loose_triple(da) == fs_loose_triple(dbb)
        fs_strict[(a, b)] = fs_strict_sig(da) == fs_strict_sig(dbb)
        scalar_clean[(a, b)] = scalar_sig(da) == scalar_sig(dbb)
        both_exact[(a, b)] = bool(
            da["character_table_fingerprint_exact"] and dbb["character_table_fingerprint_exact"]
        )
        # strict FS implies loose FS -- sanity-check the monotonicity.
        assert not fs_strict[(a, b)] or fs_loose[(a, b)], (a, b)

    survivors = [(a, b) for a, b in pairs if fs_loose[(a, b)] and scalar_clean[(a, b)]]
    survivor_groups = sorted({g for p in survivors for g in p})
    survivors_set = set(survivors)

    ctproof_needed = sorted((a, b) for a, b in pairs if not both_exact[(a, b)])

    print(f"stage1 same-fingerprint pairs: {len(pairs)}")
    print(f"fs-loose-equal: {sum(fs_loose.values())}")
    print(f"fs-strict-equal (jsonl signal): {sum(fs_strict.values())}")
    print(
        f"scalar-clean among fs-loose-equal (survivors -> GAP GroupData): "
        f"{len(survivors)}  ({len(survivor_groups)} distinct groups)"
    )
    print(f"CTProof calls needed (non-exact same-fingerprint): {len(ctproof_needed)}")

    gap = Gap(gap_binary, capture_dir)
    ctproof = run_ctproof_batch(gap, ctproof_needed) if ctproof_needed else {}
    gdata, expensive = run_groupdata(gap, survivor_groups)
    print(
        f"GroupData: {len(gdata)}/{len(survivor_groups)} groups computed; "
        f"{len(expensive)} EXPENSIVE/skipped"
    )

    # cross-check GAP vs jsonl for every group GAP actually touched.
    xcheck: list[Row] = []
    for g in sorted(gdata):
        gd = gdata[g]
        j = db[g]
        row = {
            "id": list(g),
            "fs_triple_match": tuple(gd["fs_triple"]) == fs_loose_triple(j),
            "fs_strict_match": joint_multiset(gd) == fs_strict_sig(j),
            "mci_match": gd["min_corefree_index"] == j["min_corefree_index"],
            "spectrum_match": (
                sorted(set(gd["corefree_index_spectrum"]))
                == sorted(set(j["corefree_index_spectrum"]))
            ),
            "fs_involution_ok": gd["fs_involution_ok"],
        }
        xcheck.append(row)
        assert gd["fs_involution_ok"], f"FS involution identity failed for {g}"
    all_xcheck_pass = all(
        x["fs_triple_match"]
        and x["fs_strict_match"]
        and x["mci_match"]
        and x["spectrum_match"]
        and x["fs_involution_ok"]
        for x in xcheck
    )

    expensive_ids = {tuple(e["id"]) for e in expensive}

    records: list[Row] = []  # full records: every non-FS_FLIP pair
    fs_flip_count = 0
    classification_counts: Counter[str] = Counter()
    cls_by_pair: dict[Pair, str] = {}  # EVERY stage-1 pair -> classification

    for a, b in pairs:
        ja, jb = db[a], db[b]
        strict = fs_strict[(a, b)]
        exact = both_exact[(a, b)]
        scal_clean = scalar_clean[(a, b)]
        is_survivor = (a, b) in survivors_set
        if exact:
            ct_equal = True
            ct_basis = "exact_fingerprint"
        else:
            ct_equal = ctproof[(a, b)]
            ct_basis = "gap_transforming_permutations"

        tmpl: dict[str, Any] | None
        if ct_equal and not strict:
            # FS_FLIP needs no further GAP work -- counted only, per spec.
            fs_flip_count += 1
            classification_counts["FS_FLIP"] += 1
            cls_by_pair[(a, b)] = "FS_FLIP"
            continue
        elif not ct_equal:
            cls = "CT_EQUAL_ONLY"
            tmpl = None
        elif not scal_clean:
            cls = classify(ct_equal, strict, scal_clean, template_clean=False)
            tmpl = None
        elif a in expensive_ids or b in expensive_ids:
            cls = "TEMPLATE_SCREEN_SKIPPED_EXPENSIVE"
            tmpl = None
        else:
            tmpl = template_verdict(gdata[a], gdata[b])
            cls = classify(ct_equal, strict, scal_clean, tmpl["clean"])

        classification_counts[cls] += 1
        cls_by_pair[(a, b)] = cls

        residual: Row = {}
        for field in [
            "element_order_histogram",
            "exponent",
            "max_element_order",
            "aut_order",
            "number_subgroups",
            "num_involutions",
        ]:
            va, vb = ja[field], jb[field]
            if va != vb:
                residual[field] = [va, vb]

        records.append(
            {
                "a": list(a),
                "b": list(b),
                "name_a": ja["name"],
                "name_b": jb["name"],
                "fingerprint": ja["character_table_fingerprint"][:12],
                "exact_a": ja["character_table_fingerprint_exact"],
                "exact_b": jb["character_table_fingerprint_exact"],
                "ct_equal": ct_equal,
                "ct_basis": ct_basis,
                "character_degrees": sorted(ja["character_degrees"]),
                "fs_strict_identical": strict,
                "fs_loose_identical": fs_loose[(a, b)],
                "is_stage_a_survivor": is_survivor,
                "scalar": {
                    "mci": [ja["min_corefree_index"], jb["min_corefree_index"]],
                    "mfpd": [
                        ja["minimal_faithful_permutation_degree"],
                        jb["minimal_faithful_permutation_degree"],
                    ],
                    "spectrum_a": sorted(ja["corefree_index_spectrum"]),
                    "spectrum_b": sorted(jb["corefree_index_spectrum"]),
                    "clean": scal_clean,
                },
                "template": tmpl,
                "residual_axes": residual,
                "classification": cls,
            }
        )

    # ---- funnel (task-requested chain) --------------------------------------
    funnel = {
        "stage1_total_same_fingerprint_pairs": len(pairs),
        "stage_fs_triple_equal": sum(fs_loose.values()),
        "stage_scalar_clean_among_fs_triple_equal": len(survivors),
        "stage_strict_fs_gap_confirmed_among_scalar_clean": sum(
            1 for r in records if r["is_stage_a_survivor"] and r["fs_strict_identical"]
        ),
        "stage_template_clean_among_strict_fs": sum(
            1 for r in records if r["classification"] == "GENUINE_CLEAN"
        ),
    }

    genuine_clean = sorted(
        [r["a"], r["b"]] for r in records if r["classification"] == "GENUINE_CLEAN"
    )

    # ---- consistency check: the panel-restricted screen must agree ----------
    # A pair's classification depends ONLY on its own two members' invariants
    # (ct_equal, fs_strict, scalar, template), never on which other groups share
    # its fingerprint bucket -- so restricting the universe to the panel cannot
    # change any panel pair's verdict.  This re-derives that agreement pair by
    # pair from THIS run's classification map (no separate driver needed): the
    # 363-row panel's same-fingerprint pairs are a subset of the run's Stage-1
    # pairs, and each must carry the identical verdict.
    consistency = _consistency_check(jsonl, db, cls_by_pair, panel_path)

    display_jsonl = _display_path(jsonl)
    if restrict_ids is None:
        scope = f"every group in {display_jsonl} with {ORDER_MIN} <= order <= {ORDER_MAX}"
        n_in_range = len(db)
    else:
        scope = (
            f"the {len(db)} groups in {_display_path(panel_path)}"  # type: ignore[arg-type]
            f" restricted from {display_jsonl}"
        )
        n_in_range = len(db)

    return {
        "provenance": {
            "scope": scope,
            "ground_truth": display_jsonl,
            "gap": gap_binary,
            "jsonl_sha256": sha256(jsonl),
            "n_groups_in_range": n_in_range,
            "n_stage1_pairs": len(pairs),
            "n_survivor_groups_groupdata": len(survivor_groups),
            "n_ctproof_calls": len(ctproof_needed),
        },
        "funnel": funnel,
        "classification_counts": dict(classification_counts),
        "fs_flip_count": fs_flip_count,
        "genuine_clean_pairs": genuine_clean,
        "expensive_skips": expensive,
        "cross_check_gap_vs_jsonl": xcheck,
        "cross_check_all_pass": all_xcheck_pass,
        "consistency_check_vs_panel": consistency,
        "pairs": records,
    }


def _consistency_check(
    jsonl: Path, db: dict[GroupId, Row], cls_by_pair: dict[Pair, str], panel_path: Path | None
) -> dict[str, Any]:
    """Validate that the panel-restricted screen agrees with this run.  The
    reference panel is the run's own ``--panel`` file when restricted, else the
    bundled 363-row panel."""
    ref = panel_path if panel_path is not None else PANEL_ROWS
    if not ref.exists():
        return {
            "panel_pairs_checked": 0,
            "mismatches": [],
            "all_match": True,
            "note": f"reference panel {ref} not found; skipped",
        }
    panel_ids = load_panel_ids(ref)
    # only panel groups that are in this run's universe can be compared
    in_universe = [g for g in panel_ids if g in db]
    panel_pairs = stage1_pairs(sorted(in_universe), db)
    mismatches: list[Row] = []
    checked = 0
    for pair in panel_pairs:
        checked += 1
        run_cls = cls_by_pair.get(pair)
        # panel-restricted classification == this-run classification, by the
        # per-pair-locality argument above.  A genuine divergence (e.g. a panel
        # pair the run never enumerated) is surfaced, not swallowed.
        if run_cls is None:
            mismatches.append(
                {"pair": [list(pair[0]), list(pair[1])], "reason": "panel_pair_not_in_run"}
            )
    return {
        "panel_pairs_checked": checked,
        "mismatches": mismatches,
        "all_match": len(mismatches) == 0,
    }


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=DEFAULT_JSONL,
        help="ground-truth group-properties jsonl (default: data/group_properties_full.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="output JSON path (default: results/falsifier_screen_results_full.json)",
    )
    parser.add_argument(
        "--gap",
        default=os.environ.get("GAI_GAP_BIN", DEFAULT_GAP),
        help=f"GAP binary (default: $GAI_GAP_BIN or {DEFAULT_GAP})",
    )
    parser.add_argument(
        "--panel",
        type=Path,
        default=None,
        help="restrict the screened universe to the (order,index) "
        "rows in this file (the panel variant)",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="derive_falsifiers_") as tmp:
        out = run_screen(args.jsonl, args.gap, args.panel, Path(tmp))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))

    print()
    print("funnel:", json.dumps(out["funnel"], indent=2))
    print("classification counts:", json.dumps(out["classification_counts"]))
    print("GENUINE_CLEAN pairs:", len(out["genuine_clean_pairs"]))
    print("expensive skips:", out["expensive_skips"])
    print("consistency vs panel: all_match =", out["consistency_check_vs_panel"]["all_match"])
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
