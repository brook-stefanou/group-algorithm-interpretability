"""Tests for ``scripts/derive_falsifiers.py`` -- the falsifier screen that
defines claim C1.

Two layers:

* Always-run, offline: schema of the committed results file, the pure
  classification logic on synthetic fixtures, deterministic pair ordering, and
  sha256 pinning of the committed inputs/outputs. These need neither GAP nor the
  (gitignored) ground-truth jsonl, so they run in CI.
* GAP-required, skipped without a GAP binary (mirrors the Sage skip in
  ``tests/test_sage_backend.py``): a tiny end-to-end screen over a handful of
  small groups asserting known verdicts. The offline gate stays green with these
  skipped.

``scripts/derive_falsifiers.py`` is an executable entry point, not part of the
installed package, and imports only the standard library at module load (no
GAP), so it is loaded here by file path -- mirroring
``tests/test_enumerate_groups_pipeline.py``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

_REPO = Path(__file__).resolve().parent.parent
_SCRIPT_PATH = _REPO / "scripts" / "derive_falsifiers.py"
_RESULTS = _REPO / "results" / "falsifier_screen_results_full.json"
_PANEL_ROWS = _REPO / "results" / "falsifier_panel_rows.json"

# Pinned provenance, recorded in results/README.md and the committed results.
_JSONL_SHA256 = "b3071828c7d13eceb90453ad24c87e89adf402784f4a3c11d1d77e7e34b0536b"
_PANEL_ROWS_SHA256 = "bc4617e7afcd1fb6db98df2f7cbc57655c519dbed1bd47f72e8b9017e870a1ec"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("derive_falsifiers", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


df = _load_module()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gap_binary() -> str | None:
    """The GAP binary the screen would use, if it is actually runnable here."""
    candidate = os.environ.get("GAI_GAP_BIN", df.DEFAULT_GAP)
    if Path(candidate).exists() and os.access(candidate, os.X_OK):
        return candidate
    found = shutil.which("gap")
    return found


# --------------------------------------------------------------------------- #
# Offline: committed results schema                                           #
# --------------------------------------------------------------------------- #
def test_committed_results_file_exists() -> None:
    assert _RESULTS.exists(), "committed falsifier results must ship in results/"
    assert _PANEL_ROWS.exists(), "committed panel-rows input must ship in results/"


def test_committed_results_top_level_schema() -> None:
    data = json.loads(_RESULTS.read_text())
    assert list(data.keys()) == [
        "provenance",
        "funnel",
        "classification_counts",
        "fs_flip_count",
        "genuine_clean_pairs",
        "expensive_skips",
        "cross_check_gap_vs_jsonl",
        "cross_check_all_pass",
        "consistency_check_vs_panel",
        "pairs",
    ]
    prov = data["provenance"]
    assert prov["jsonl_sha256"] == _JSONL_SHA256
    assert prov["gap"] == df.DEFAULT_GAP
    assert prov["n_groups_in_range"] == 6958


def test_committed_results_invariants() -> None:
    data = json.loads(_RESULTS.read_text())
    # every screen cross-check passed, and the panel-restricted screen agrees.
    assert data["cross_check_all_pass"] is True
    assert data["consistency_check_vs_panel"]["all_match"] is True
    assert data["consistency_check_vs_panel"]["mismatches"] == []
    assert data["expensive_skips"] == []
    # C1's family: 370 GENUINE_CLEAN pairs, matching the classification count.
    assert data["classification_counts"]["GENUINE_CLEAN"] == 370
    assert len(data["genuine_clean_pairs"]) == 370
    assert data["classification_counts"]["FS_FLIP"] == data["fs_flip_count"]


def test_committed_genuine_clean_pairs_wellformed_and_sorted() -> None:
    data = json.loads(_RESULTS.read_text())
    pairs = data["genuine_clean_pairs"]
    for pair in pairs:
        assert len(pair) == 2
        for member in pair:
            assert len(member) == 2 and all(isinstance(x, int) for x in member)
    assert pairs == sorted(pairs), "genuine_clean_pairs must be deterministically sorted"
    # a couple of anchor pairs the docs cite must be present.
    assert [[27, 3], [27, 4]] in pairs
    assert [[64, 74], [64, 80]] in pairs


def test_committed_pair_records_schema() -> None:
    data = json.loads(_RESULTS.read_text())
    required = {
        "a",
        "b",
        "name_a",
        "name_b",
        "fingerprint",
        "exact_a",
        "exact_b",
        "ct_equal",
        "ct_basis",
        "character_degrees",
        "fs_strict_identical",
        "fs_loose_identical",
        "is_stage_a_survivor",
        "scalar",
        "template",
        "residual_axes",
        "classification",
    }
    genuine = [r for r in data["pairs"] if r["classification"] == "GENUINE_CLEAN"]
    assert len(genuine) == 370
    for rec in genuine:
        assert required <= set(rec.keys())
        assert rec["ct_equal"] is True
        assert rec["fs_strict_identical"] is True
        assert rec["scalar"]["clean"] is True
        assert rec["template"]["clean"] is True


# --------------------------------------------------------------------------- #
# Offline: sha256 pinning                                                     #
# --------------------------------------------------------------------------- #
def test_panel_rows_input_sha256_pinned() -> None:
    assert _sha256(_PANEL_ROWS) == _PANEL_ROWS_SHA256


def test_committed_results_pins_jsonl_hash() -> None:
    data = json.loads(_RESULTS.read_text())
    assert data["provenance"]["jsonl_sha256"] == _JSONL_SHA256


# --------------------------------------------------------------------------- #
# Offline: classification logic on synthetic fixtures                         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("ct_equal", "fs_strict", "scalar_clean", "template_clean", "expected"),
    [
        (False, True, True, True, "CT_EQUAL_ONLY"),
        (False, False, False, False, "CT_EQUAL_ONLY"),
        (True, False, True, True, "FS_FLIP"),  # FS-flip
        (True, True, False, True, "FS_IDENTICAL_COSET_CONFOUNDED"),  # scalar-fail
        (True, True, True, False, "FS_IDENTICAL_COSET_CONFOUNDED"),  # template-fail
        (True, True, True, True, "GENUINE_CLEAN"),  # FS-identical, fully clean
    ],
)
def test_classify_rule(
    ct_equal: bool, fs_strict: bool, scalar_clean: bool, template_clean: bool, expected: str
) -> None:
    assert df.classify(ct_equal, fs_strict, scalar_clean, template_clean) == expected


def _row(degrees: list[int], indicator: list[int], **over: object) -> dict[str, object]:
    row: dict[str, object] = {
        "character_degrees": degrees,
        "indicator_vector": indicator,
        "fs_real_count": sum(1 for v in indicator if v == 1),
        "fs_complex_count": sum(1 for v in indicator if v == 0),
        "fs_quaternionic_count": sum(1 for v in indicator if v == -1),
        "min_corefree_index": 3,
        "minimal_faithful_permutation_degree": 3,
        "corefree_index_spectrum": [3, 9],
    }
    row.update(over)
    return row


def test_fs_strict_sig_identical_vs_flip() -> None:
    a = _row([1, 1, 1, 2], [1, 1, 1, 1])
    identical = _row([1, 1, 1, 2], [1, 1, 1, 1])
    # an FS-flip: same degrees, one real irrep becomes quaternionic.
    flip = _row([1, 1, 1, 2], [1, 1, 1, -1])
    assert df.fs_strict_sig(a) == df.fs_strict_sig(identical)
    assert df.fs_strict_sig(a) != df.fs_strict_sig(flip)
    # ordering of the unsorted degree/indicator lists must not matter.
    reordered = _row([2, 1, 1, 1], [1, 1, 1, 1])
    assert df.fs_strict_sig(a) == df.fs_strict_sig(reordered)


def test_scalar_sig_clean_vs_fail() -> None:
    a = _row([1, 2], [1, 1], min_corefree_index=3, corefree_index_spectrum=[3, 9])
    clean = _row([1, 2], [1, 1], min_corefree_index=3, corefree_index_spectrum=[9, 3])
    scalar_fail = _row([1, 2], [1, 1], min_corefree_index=27, corefree_index_spectrum=[27, 81])
    assert df.scalar_sig(a) == df.scalar_sig(clean)  # spectrum sorted before compare
    assert df.scalar_sig(a) != df.scalar_sig(scalar_fail)


def _gd(templates: list[dict[str, object]]) -> dict[str, object]:
    return {"templates": templates}


def test_template_verdict_clean_vs_template_fail() -> None:
    tmpl_a = _gd([{"idx": 3, "support": 2, "tmpl": [[[1, 1], 1], [[2, 1], 1]]}])
    tmpl_identical = _gd([{"idx": 3, "support": 2, "tmpl": [[[1, 1], 1], [[2, 1], 1]]}])
    # template-fail: a different occupancy support at the same index.
    tmpl_fail = _gd([{"idx": 3, "support": 1, "tmpl": [[[1, 1], 1]]}])

    assert df.template_verdict(tmpl_a, tmpl_identical)["clean"] is True
    verdict = df.template_verdict(tmpl_a, tmpl_fail)
    assert verdict["clean"] is False
    assert verdict["differing_index_detail"], "a template difference must be reported"


# --------------------------------------------------------------------------- #
# Offline: deterministic pair ordering                                        #
# --------------------------------------------------------------------------- #
def test_stage1_pairs_sorted_and_order_independent() -> None:
    db = {
        (32, 20): {"character_table_fingerprint": "F1"},
        (32, 18): {"character_table_fingerprint": "F1"},
        (27, 4): {"character_table_fingerprint": "F2"},
        (27, 3): {"character_table_fingerprint": "F2"},
        (50, 1): {"character_table_fingerprint": "solo"},
    }
    ids = list(db)
    pairs = df.stage1_pairs(ids, db)
    # only same-fingerprint pairs, emitted sorted, singletons excluded.
    assert pairs == [((27, 3), (27, 4)), ((32, 18), (32, 20))]
    assert pairs == sorted(pairs)
    # order of the input id list must not change the result.
    assert df.stage1_pairs(list(reversed(ids)), db) == pairs


def test_load_panel_ids_accepts_both_shapes(tmp_path: Path) -> None:
    wrapped = tmp_path / "wrapped.json"
    wrapped.write_text(json.dumps({"rows": [{"order": 27, "index": 3}, {"order": 27, "index": 4}]}))
    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps([{"order": 27, "index": 3}, {"order": 27, "index": 4}]))
    assert df.load_panel_ids(wrapped) == [(27, 3), (27, 4)]
    assert df.load_panel_ids(bare) == [(27, 3), (27, 4)]


# --------------------------------------------------------------------------- #
# GAP-required: end-to-end known verdicts                                     #
# --------------------------------------------------------------------------- #
_GAP = _gap_binary()
_JSONL = df.DEFAULT_JSONL


@pytest.mark.skipif(
    _GAP is None or not _JSONL.exists(),
    reason=(
        "GAP binary and the ground-truth data/group_properties_full.jsonl are "
        "both required to run the screen end-to-end (neither is present in CI). "
        "Set $GAI_GAP_BIN to a GAP 4.x binary and provide the jsonl to exercise "
        "these. The committed results file is schema-tested offline above."
    ),
)
def test_end_to_end_small_group_verdicts(tmp_path: Path) -> None:
    panel = tmp_path / "mini_panel.json"
    panel.write_text(
        json.dumps(
            {
                "rows": [
                    {"order": 27, "index": 3},
                    {"order": 27, "index": 4},
                    {"order": 32, "index": 18},
                    {"order": 32, "index": 20},
                ]
            }
        )
    )
    capture = tmp_path / "gap_capture"
    capture.mkdir()
    assert _GAP is not None  # narrowed for mypy; guarded by the skipif above
    out = df.run_screen(_JSONL, _GAP, panel, capture)

    verdicts = {(tuple(r["a"]), tuple(r["b"])): r["classification"] for r in out["pairs"]}
    # (27,3)/(27,4) = 3^{1+2}_+ / 3^{1+2}_- : the cheapest GENUINE_CLEAN pair.
    assert verdicts[((27, 3), (27, 4))] == "GENUINE_CLEAN"
    assert [[27, 3], [27, 4]] in out["genuine_clean_pairs"]
    # (32,18)/(32,20) share a character table but are an FS flip.
    assert out["classification_counts"]["FS_FLIP"] == 1
    assert [[32, 18], [32, 20]] not in out["genuine_clean_pairs"]
    # every GAP/jsonl cross-check and the FS-involution identity held.
    assert out["cross_check_all_pass"] is True


def test_end_to_end_invocation_via_cli(tmp_path: Path) -> None:
    if _GAP is None or not _JSONL.exists():
        pytest.skip("GAP binary and ground-truth jsonl required")
    panel = tmp_path / "mini_panel.json"
    panel.write_text(json.dumps({"rows": [{"order": 27, "index": 3}, {"order": 27, "index": 4}]}))
    output = tmp_path / "out.json"
    env = dict(os.environ, GAI_GAP_BIN=_GAP)
    subprocess.run(
        [
            sys.executable,
            str(_SCRIPT_PATH),
            "--panel",
            str(panel),
            "--output",
            str(output),
        ],
        check=True,
        cwd=_REPO,
        env=env,
    )
    data = json.loads(output.read_text())
    assert data["genuine_clean_pairs"] == [[[27, 3], [27, 4]]]
