"""Focused tests for ``scripts/dedup_runs.py`` (the non-destructive
canonical-run index and rerun-agreement check).

Offline and network-free: run directories are synthesised on disk with tiny
``resolved_config.yaml``/``manifest.yaml``/``selection.json`` fixtures. The
script is an executable entry point loaded by file path (mirrors
``tests/test_grok_verdict.py`` and ``test_backfill_runs.py``), since
``scripts/`` is not a package.

Covers exactly the three cases the task calls for: a seed with two completed
reruns that agree, a seed whose reruns disagree on grokked-vs-censored, and a
singleton (no duplicate) -- plus the canonical-selection tiebreak rule and
the non-destructive nature of the index (no file under ``runs/`` is ever
written to or removed).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "dedup_runs.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("dedup_runs", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["dedup_runs"] = module
    spec.loader.exec_module(module)
    return module


dedup_runs = _load_script()


# ---------------------------------------------------------------------------
# fixtures: run directories on disk
# ---------------------------------------------------------------------------


def _write_run(
    root: Path,
    run_id: str,
    *,
    order: int,
    index: int,
    width: int,
    epochs: int,
    seed: int,
    status: str = "completed",
    completed_at: str | None,
    selection: dict[str, Any] | None,
) -> Path:
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(
            {
                "seed": seed,
                "data": {"group": {"order": order, "index": index}},
                "model": {"d_model": width},
                "optim": {"epochs": epochs},
            }
        )
    )
    (run_dir / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "run_id": run_id,
                "status": status,
                "completed_at": completed_at,
                "dataset": {"leakage": {"generalize_metric": "unleaked_accuracy"}},
            }
        )
    )
    if selection is not None:
        (run_dir / "selection.json").write_text(json.dumps(selection))
    return run_dir


def _flat_grokked(epoch: int) -> dict[str, Any]:
    return {
        "censored_unleaked": False,
        "unleaked_cross_epoch": epoch,
        "censored_leaked": False,
        "leaked_cross_epoch": epoch - 500,
    }


def _flat_censored() -> dict[str, Any]:
    return {
        "censored_unleaked": True,
        "unleaked_cross_epoch": None,
        "censored_leaked": False,
        "leaked_cross_epoch": 40000,
    }


# ---------------------------------------------------------------------------
# choose_canonical: the tiebreak rule itself
# ---------------------------------------------------------------------------


def test_choose_canonical_prefers_completed_status():
    infos = [
        {"run_id": "b-later", "status": "failed", "completed_at": None},
        {"run_id": "a-earlier", "status": "completed", "completed_at": "2026-01-01T00:00:00+00:00"},
    ]
    canonical, dropped = dedup_runs.choose_canonical(infos)
    assert canonical == "a-earlier"
    assert dropped == ["b-later"]


def test_choose_canonical_earliest_completed_at_among_completed():
    infos = [
        {"run_id": "later", "status": "completed", "completed_at": "2026-01-02T00:00:00+00:00"},
        {"run_id": "earlier", "status": "completed", "completed_at": "2026-01-01T00:00:00+00:00"},
    ]
    canonical, dropped = dedup_runs.choose_canonical(infos)
    assert canonical == "earlier"
    assert dropped == ["later"]


def test_choose_canonical_falls_back_to_run_id_when_completed_at_missing():
    infos = [
        {"run_id": "2026-07-20_999999_core_zzz", "status": "completed", "completed_at": None},
        {"run_id": "2026-07-20_100000_core_aaa", "status": "completed", "completed_at": None},
    ]
    canonical, dropped = dedup_runs.choose_canonical(infos)
    assert canonical == "2026-07-20_100000_core_aaa"
    assert dropped == ["2026-07-20_999999_core_zzz"]


def test_choose_canonical_singleton_drops_nothing():
    canonical, dropped = dedup_runs.choose_canonical(
        [{"run_id": "only", "status": "completed", "completed_at": "2026-01-01T00:00:00+00:00"}]
    )
    assert canonical == "only"
    assert dropped == []


# ---------------------------------------------------------------------------
# classify_agreement: full agreement, epoch-only, and label disagreement
# ---------------------------------------------------------------------------


def test_classify_agreement_full_agreement_same_label_close_epochs():
    grok_by_run = {
        "a": {"grokked": True, "censored": False, "epochs_to_grok": 10000},
        "b": {"grokked": True, "censored": False, "epochs_to_grok": 10200},
    }
    result = dedup_runs.classify_agreement(["a", "b"], grok_by_run, epoch_tolerance=1000)
    assert result["outcome"] == "agreement"
    assert result["epoch_spread"] == 200


def test_classify_agreement_censored_reruns_agree_trivially():
    grok_by_run = {
        "a": {"grokked": False, "censored": True, "epochs_to_grok": None},
        "b": {"grokked": False, "censored": True, "epochs_to_grok": None},
    }
    result = dedup_runs.classify_agreement(["a", "b"], grok_by_run, epoch_tolerance=1000)
    assert result["outcome"] == "agreement"


def test_classify_agreement_epoch_disagreement_beyond_tolerance():
    grok_by_run = {
        "a": {"grokked": True, "censored": False, "epochs_to_grok": 5000},
        "b": {"grokked": True, "censored": False, "epochs_to_grok": 40000},
    }
    result = dedup_runs.classify_agreement(["a", "b"], grok_by_run, epoch_tolerance=1000)
    assert result["outcome"] == "epoch_disagreement"
    assert result["epoch_spread"] == 35000


def test_classify_agreement_label_disagreement():
    grok_by_run = {
        "a": {"grokked": True, "censored": False, "epochs_to_grok": 26000},
        "b": {"grokked": False, "censored": True, "epochs_to_grok": None},
    }
    result = dedup_runs.classify_agreement(["a", "b"], grok_by_run, epoch_tolerance=1000)
    assert result["outcome"] == "label_disagreement"
    assert result["epoch_spread"] is None


def test_classify_agreement_missing_data():
    grok_by_run = {"a": {"grokked": True, "censored": False, "epochs_to_grok": 100}, "b": None}
    result = dedup_runs.classify_agreement(["a", "b"], grok_by_run, epoch_tolerance=1000)
    assert result["outcome"] == "missing_data"


# ---------------------------------------------------------------------------
# build_index / main: end-to-end over synthesised run directories
# ---------------------------------------------------------------------------


def test_build_index_three_cases(tmp_path):
    """A seed with two agreeing completed reruns, a seed whose reruns
    disagree on grok/censor, and a singleton -- exactly the fixture set the
    task calls for."""
    # Seed 0: two completed reruns, both grok, close epochs -> agreement.
    _write_run(
        tmp_path,
        "2026-01-01_000000_000000_core_aaa",
        order=32,
        index=18,
        width=128,
        epochs=60000,
        seed=0,
        completed_at="2026-01-01T00:10:00+00:00",
        selection=_flat_grokked(27000),
    )
    _write_run(
        tmp_path,
        "2026-01-01_010000_000000_core_bbb",
        order=32,
        index=18,
        width=128,
        epochs=60000,
        seed=0,
        completed_at="2026-01-01T01:10:00+00:00",
        selection=_flat_grokked(27300),
    )

    # Seed 1: two completed reruns that disagree on grok vs censor.
    _write_run(
        tmp_path,
        "2026-01-01_000000_000000_core_ccc",
        order=32,
        index=18,
        width=128,
        epochs=60000,
        seed=1,
        completed_at="2026-01-01T00:10:00+00:00",
        selection=_flat_grokked(26000),
    )
    _write_run(
        tmp_path,
        "2026-01-01_010000_000000_core_ddd",
        order=32,
        index=18,
        width=128,
        epochs=60000,
        seed=1,
        completed_at="2026-01-01T01:10:00+00:00",
        selection=_flat_censored(),
    )

    # Seed 2: a singleton, no rerun.
    _write_run(
        tmp_path,
        "2026-01-01_000000_000000_core_eee",
        order=32,
        index=18,
        width=128,
        epochs=60000,
        seed=2,
        completed_at="2026-01-01T00:10:00+00:00",
        selection=_flat_grokked(20000),
    )

    run_dirs = dedup_runs.discover_run_dirs(tmp_path)
    index, rerun_agreement, n_unresolvable = dedup_runs.build_index(run_dirs, epoch_tolerance=1000)

    assert n_unresolvable == 0
    assert len(index) == 3  # three distinct (order, index, width, epochs, seed) keys

    seed0_key = (32, 18, 128, 60000, 0)
    seed1_key = (32, 18, 128, 60000, 1)
    seed2_key = (32, 18, 128, 60000, 2)

    # Seed 0: earliest completed_at wins; the later rerun is dropped, not deleted.
    assert index[seed0_key]["canonical_run_id"] == "2026-01-01_000000_000000_core_aaa"
    assert index[seed0_key]["dropped_run_ids"] == ["2026-01-01_010000_000000_core_bbb"]
    assert index[seed0_key]["n_total"] == 2

    # Seed 1: same tiebreak rule applies regardless of the grok/censor outcome.
    assert index[seed1_key]["canonical_run_id"] == "2026-01-01_000000_000000_core_ccc"

    # Seed 2: singleton, nothing dropped.
    assert index[seed2_key]["canonical_run_id"] == "2026-01-01_000000_000000_core_eee"
    assert index[seed2_key]["dropped_run_ids"] == []

    # Rerun agreement: one full agreement (seed 0), one label disagreement (seed 1).
    assert rerun_agreement["n_duplicate_keys"] == 2
    assert rerun_agreement["n_full_agreement"] == 1
    assert rerun_agreement["n_label_disagreement"] == 1
    assert rerun_agreement["n_epoch_disagreement"] == 0

    disagreement = rerun_agreement["label_disagreements"][0]
    assert disagreement["key"] == {
        "order": 32,
        "index": 18,
        "width": 128,
        "epochs": 60000,
        "seed": 1,
    }
    outcomes = {r["run_id"]: r["censored"] for r in disagreement["runs"]}
    assert outcomes == {
        "2026-01-01_000000_000000_core_ccc": False,
        "2026-01-01_010000_000000_core_ddd": True,
    }


def test_main_writes_index_and_does_not_touch_runs_root(tmp_path):
    _write_run(
        tmp_path,
        "2026-01-01_000000_000000_core_aaa",
        order=64,
        index=74,
        width=128,
        epochs=60000,
        seed=0,
        completed_at="2026-01-01T00:10:00+00:00",
        selection=_flat_grokked(5000),
    )
    _write_run(
        tmp_path,
        "2026-01-01_010000_000000_core_bbb",
        order=64,
        index=74,
        width=128,
        epochs=60000,
        seed=0,
        completed_at="2026-01-01T01:10:00+00:00",
        selection=_flat_grokked(5100),
    )
    before = {p: p.read_bytes() if p.is_file() else None for p in tmp_path.rglob("*")}

    out_path = tmp_path.parent / "canonical_runs_out.json"
    rc = dedup_runs.main([str(tmp_path), "--out", str(out_path)])
    assert rc == 0

    # runs_root is byte-for-byte unchanged: no file added, removed, or edited.
    after = {p: p.read_bytes() if p.is_file() else None for p in tmp_path.rglob("*")}
    assert before == after

    payload = json.loads(out_path.read_text())
    assert payload["n_run_dirs"] == 2
    assert payload["n_distinct_keys"] == 1
    assert payload["n_duplicate_keys"] == 1
    assert payload["n_dropped_duplicate_dirs"] == 1
    assert payload["grok_totals_raw"] == {"grokked": 2, "censored": 0, "missing": 0}
    assert payload["grok_totals_canonical"] == {"grokked": 1, "censored": 0, "missing": 0}
    key = next(iter(payload["canonical_runs"].values()))
    assert key["canonical_run_id"] == "2026-01-01_000000_000000_core_aaa"
    assert key["dropped_run_ids"] == ["2026-01-01_010000_000000_core_bbb"]


def test_main_missing_runs_root_returns_error(tmp_path):
    rc = dedup_runs.main([str(tmp_path / "does-not-exist")])
    assert rc == 1
