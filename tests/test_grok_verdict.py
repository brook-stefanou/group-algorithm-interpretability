"""Focused tests for ``scripts/grok_verdict.py`` (the per-cell grok/censoring
verdict).

Offline and network-free: run directories are synthesised on disk with tiny
``resolved_config.yaml``/``manifest.yaml``/``selection.json`` fixtures covering
both ``selection.json`` schemas (flat and ``categories``) and both outcomes
(grokked and censored). The script is an executable entry point loaded by file
path (mirrors ``tests/test_backfill_runs.py`` and ``test_measure_probes.py``),
since ``scripts/`` is not a package.

The decisive property: the verdict this script derives comes from
``publish._grok_fields_from_selection`` (imported, not reimplemented), so a
flat-schema grokked run, a categories-schema censored run, and the raw-fallback
leaked-metric case all resolve exactly as the W&B publisher would resolve them.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "grok_verdict.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("grok_verdict", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["grok_verdict"] = module
    spec.loader.exec_module(module)
    return module


grok_verdict = _load_script()


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
    selection: dict[str, Any] | None,
    generalize_metric: str = "unleaked_accuracy",
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
                "experiment": {"name": "core"},
            }
        )
    )
    (run_dir / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "run_id": run_id,
                "status": "completed",
                "dataset": {"leakage": {"generalize_metric": generalize_metric}},
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


def _categories_grokked(epoch: int) -> dict[str, Any]:
    return {
        "censored": False,
        "unleaked_metric_key": "val/unleaked_accuracy",
        "leaked_metric_key": "val/accuracy",
        "categories": {
            "unleaked_onset": {"epoch": epoch, "filename": f"step_{epoch}.pt"},
            "leaked_onset": {"epoch": epoch - 500, "filename": "step.pt"},
        },
    }


def _categories_censored() -> dict[str, Any]:
    return {
        "censored": True,
        "unleaked_metric_key": "val/unleaked_accuracy",
        "leaked_metric_key": "val/accuracy",
        "categories": {"unleaked_onset": None, "leaked_onset": None},
    }


# ---------------------------------------------------------------------------
# per-run grok derivation reuses publish._grok_fields_from_selection
# ---------------------------------------------------------------------------


def test_flat_schema_grokked_and_censored(tmp_path):
    grok_dir = _write_run(
        tmp_path,
        "flat-grok",
        order=64,
        index=74,
        width=128,
        epochs=60000,
        seed=0,
        selection=_flat_grokked(4800),
    )
    cens_dir = _write_run(
        tmp_path,
        "flat-cens",
        order=64,
        index=74,
        width=128,
        epochs=60000,
        seed=1,
        selection=_flat_censored(),
    )
    assert grok_verdict.run_grok_fields(grok_dir) == {
        "grokked": True,
        "censored": False,
        "epochs_to_grok": 4800,
    }
    assert grok_verdict.run_grok_fields(cens_dir) == {
        "grokked": False,
        "censored": True,
        "epochs_to_grok": None,
    }


def test_categories_schema_grokked_and_censored(tmp_path):
    grok_dir = _write_run(
        tmp_path,
        "cat-grok",
        order=32,
        index=18,
        width=128,
        epochs=60000,
        seed=0,
        selection=_categories_grokked(17500),
    )
    cens_dir = _write_run(
        tmp_path,
        "cat-cens",
        order=32,
        index=18,
        width=128,
        epochs=60000,
        seed=1,
        selection=_categories_censored(),
    )
    assert grok_verdict.run_grok_fields(grok_dir)["epochs_to_grok"] == 17500
    assert grok_verdict.run_grok_fields(grok_dir)["grokked"] is True
    assert grok_verdict.run_grok_fields(cens_dir)["censored"] is True


def test_raw_fallback_uses_leaked_metric(tmp_path):
    """A run whose manifest declares the raw-accuracy fallback (empty unleaked
    subset) is scored on the leaked signal -- mirrors publish.py. Here the
    unleaked onset never lands but the leaked one does, so it must count as
    grokked, not censored."""
    run_dir = _write_run(
        tmp_path,
        "raw-fallback",
        order=27,
        index=3,
        width=256,
        epochs=60000,
        seed=0,
        selection={
            # ship_runs' top-level `censored` is leaked-based (leaked_onset is
            # None); here the leaked onset landed, so it is False even though
            # the unleaked onset never landed.
            "censored": False,
            "unleaked_metric_key": "val/unleaked_accuracy",
            "leaked_metric_key": "val/accuracy",
            "categories": {
                "unleaked_onset": None,
                "leaked_onset": {"epoch": 12000, "filename": "step.pt"},
            },
        },
        generalize_metric="raw_test_accuracy",
    )
    fields = grok_verdict.run_grok_fields(run_dir)
    assert fields == {"grokked": True, "censored": False, "epochs_to_grok": 12000}


def test_missing_selection_returns_none(tmp_path):
    run_dir = _write_run(
        tmp_path,
        "no-selection",
        order=8,
        index=3,
        width=32,
        epochs=60000,
        seed=0,
        selection=None,
    )
    assert grok_verdict.run_grok_fields(run_dir) is None


# ---------------------------------------------------------------------------
# cell aggregation
# ---------------------------------------------------------------------------


def test_aggregate_groups_by_cell_and_counts(tmp_path):
    # Cell A: two grokked, one censored.
    _write_run(
        tmp_path,
        "a0",
        order=64,
        index=74,
        width=128,
        epochs=60000,
        seed=0,
        selection=_flat_grokked(4000),
    )
    _write_run(
        tmp_path,
        "a1",
        order=64,
        index=74,
        width=128,
        epochs=60000,
        seed=1,
        selection=_categories_grokked(6000),
    )
    _write_run(
        tmp_path,
        "a2",
        order=64,
        index=74,
        width=128,
        epochs=60000,
        seed=2,
        selection=_flat_censored(),
    )
    # Cell B: different width, one grokked.
    _write_run(
        tmp_path,
        "b0",
        order=64,
        index=74,
        width=256,
        epochs=60000,
        seed=0,
        selection=_flat_grokked(3000),
    )

    run_dirs = grok_verdict.discover_run_dirs(tmp_path)
    cells, n_unresolvable, n_total = grok_verdict.aggregate_runs(run_dirs)
    assert n_unresolvable == 0
    assert n_total == 4

    agg_a = cells[(64, 74, 128, 60000)]
    assert agg_a.n_grokked == 2
    assert agg_a.n_censored == 1
    assert sorted(agg_a.epochs_to_grok) == [4000.0, 6000.0]

    rec_a = grok_verdict.cell_record(agg_a, {})
    assert rec_a["n_total"] == 3
    assert rec_a["grok_fraction"] == pytest.approx(2 / 3)
    assert rec_a["epochs_to_grok"]["median"] == 5000.0


def test_rerun_seeds_counted_and_flagged(tmp_path):
    """Same nominal seed completed twice (restarted pods) is counted as two
    runs in n_total, with n_reruns exposing the duplication rather than
    silently collapsing it."""
    _write_run(
        tmp_path,
        "r0",
        order=32,
        index=18,
        width=128,
        epochs=60000,
        seed=0,
        selection=_flat_grokked(4000),
    )
    _write_run(
        tmp_path,
        "r1",
        order=32,
        index=18,
        width=128,
        epochs=60000,
        seed=0,
        selection=_flat_grokked(4200),
    )
    run_dirs = grok_verdict.discover_run_dirs(tmp_path)
    cells, _, _ = grok_verdict.aggregate_runs(run_dirs)
    rec = grok_verdict.cell_record(cells[(32, 18, 128, 60000)], {})
    assert rec["n_total"] == 2
    assert rec["n_distinct_seeds"] == 1
    assert rec["n_reruns"] == 1


def test_epochs_to_grok_interval_suppressed_below_floor(tmp_path):
    summary = grok_verdict.epochs_to_grok_summary([1000.0, 2000.0, 3000.0])
    assert summary["n"] == 3
    assert summary["median"] == 2000.0
    assert summary["p10"] is None
    assert summary["p90"] is None

    many = [float(x) for x in range(1000, 6001, 1000)]  # 6 values
    summary_many = grok_verdict.epochs_to_grok_summary(many)
    assert summary_many["p10"] is not None
    assert summary_many["p90"] is not None
    assert summary_many["p10"] < summary_many["median"] < summary_many["p90"]


def test_in_pinned_cell_list_flag(tmp_path):
    _write_run(
        tmp_path,
        "pinned",
        order=64,
        index=74,
        width=128,
        epochs=60000,
        seed=0,
        selection=_flat_grokked(4000),
    )
    _write_run(
        tmp_path,
        "extended",
        order=64,
        index=241,
        width=256,
        epochs=400000,
        seed=0,
        selection=_flat_grokked(15000),
    )
    run_dirs = grok_verdict.discover_run_dirs(tmp_path)
    cells, _, _ = grok_verdict.aggregate_runs(run_dirs)
    meta = {
        (64, 74, 128): {"name": "cell-74", "phases": {"core"}},
        (64, 241, 256): {"name": "cell-241", "phases": {"core"}},
    }
    pinned = grok_verdict.cell_record(cells[(64, 74, 128, 60000)], meta)
    extended = grok_verdict.cell_record(cells[(64, 241, 256, 400000)], meta)
    assert pinned["in_pinned_cell_list"] is True
    assert extended["in_pinned_cell_list"] is False  # 400k epochs != pinned ceiling


# ---------------------------------------------------------------------------
# campaign config parsing: tier-1 pairs and floor check
# ---------------------------------------------------------------------------


def test_find_tier1_pairs_parses_notes():
    cells = [
        {
            "order": 27,
            "index": 3,
            "width": 256,
            "name": "a",
            "note": "w256 probe -- C1 tier-1 pair with (27,4); grokked 0/52",
        },
        {
            "order": 27,
            "index": 4,
            "width": 256,
            "name": "b",
            "note": "w256 probe -- C1 tier-1 pair with (27,3); grokked 0/52",
        },
        {
            "order": 64,
            "index": 74,
            "width": 128,
            "name": "c",
            "note": "C1 tier-1 pair with (64,80) -- first clean falsifier",
        },
        {
            "order": 64,
            "index": 80,
            "width": 128,
            "name": "d",
            "note": "C1 tier-1 pair with (64,74) -- first clean falsifier",
        },
        {
            "order": 127,
            "index": 1,
            "width": 128,
            "name": "anchor",
            "note": "C3 anchor -- not a pair",
        },
    ]
    pairs = grok_verdict.find_tier1_pairs(cells)
    assert ((27, 3), (27, 4)) in pairs
    assert ((64, 74), (64, 80)) in pairs
    assert len(pairs) == 2  # de-duplicated: each pair appears once


def test_find_embedding_probe_pairs():
    cells = [
        {
            "order": 54,
            "index": 10,
            "width": 256,
            "name": "a",
            "note": "w256 probe -- C1 embedding probe with (54,11); grokked 1/25",
        },
        {
            "order": 54,
            "index": 11,
            "width": 256,
            "name": "b",
            "note": "w256 probe -- C1 embedding probe with (54,10)",
        },
    ]
    pairs = grok_verdict.find_embedding_probe_pairs(cells)
    assert pairs == [((54, 10), (54, 11))]


def test_pair_floor_flags_boundary_and_meets(tmp_path):
    # (64,74)/(64,80): both well above floor.
    for idx in (74, 80):
        for seed in range(30):
            _write_run(
                tmp_path,
                f"g{idx}-{seed}",
                order=64,
                index=idx,
                width=128,
                epochs=60000,
                seed=seed,
                selection=_flat_grokked(4000 + seed),
            )
    # (27,3)/(27,4): below floor (few grokked, rest censored).
    for idx in (3, 4):
        for seed in range(30):
            sel = _flat_grokked(10000) if seed < 5 else _flat_censored()
            _write_run(
                tmp_path,
                f"p{idx}-{seed}",
                order=27,
                index=idx,
                width=256,
                epochs=60000,
                seed=seed,
                selection=sel,
            )
    run_dirs = grok_verdict.discover_run_dirs(tmp_path)
    cells, _, _ = grok_verdict.aggregate_runs(run_dirs)
    pairs = [((27, 3), (27, 4)), ((64, 74), (64, 80))]
    records = grok_verdict.pair_records(pairs, cells, {}, pair_floor=25, label="C1 tier-1 pair")
    by_pair = {tuple(map(tuple, r["pair"])): r for r in records}
    assert by_pair[((27, 3), (27, 4))]["status"] == "characterised_boundary"
    assert by_pair[((27, 3), (27, 4))]["both_meet_floor"] is False
    assert by_pair[((64, 74), (64, 80))]["status"] == "meets_floor"
    assert by_pair[((64, 74), (64, 80))]["both_meet_floor"] is True


def test_pair_floor_uses_pinned_ceiling_cell_only(tmp_path):
    """When a group has both a pinned-ceiling cell and an extended-epochs
    follow-up, the floor check reads the pinned cell, not the extension."""
    for seed in range(30):
        _write_run(
            tmp_path,
            f"pin74-{seed}",
            order=64,
            index=74,
            width=128,
            epochs=60000,
            seed=seed,
            selection=_flat_grokked(4000) if seed < 10 else _flat_censored(),
        )
        _write_run(
            tmp_path,
            f"pin80-{seed}",
            order=64,
            index=80,
            width=128,
            epochs=60000,
            seed=seed,
            selection=_flat_grokked(4000) if seed < 10 else _flat_censored(),
        )
    # An extended follow-up where everything grokked -- must NOT be used.
    for seed in range(30):
        _write_run(
            tmp_path,
            f"ext74-{seed}",
            order=64,
            index=74,
            width=128,
            epochs=200000,
            seed=seed,
            selection=_flat_grokked(80000),
        )
    run_dirs = grok_verdict.discover_run_dirs(tmp_path)
    cells, _, _ = grok_verdict.aggregate_runs(run_dirs)
    records = grok_verdict.pair_records(
        [((64, 74), (64, 80))], cells, {}, pair_floor=25, label="C1 tier-1 pair"
    )
    # Only 10 grokked at the pinned ceiling -> below the floor of 25.
    assert records[0]["status"] == "characterised_boundary"
    assert records[0]["member_a"]["epochs"] == 60000


# ---------------------------------------------------------------------------
# end-to-end main()
# ---------------------------------------------------------------------------


def test_main_writes_json_and_reconciles(tmp_path, capsys):
    _write_run(
        tmp_path,
        "a0",
        order=64,
        index=74,
        width=128,
        epochs=60000,
        seed=0,
        selection=_flat_grokked(4000),
    )
    _write_run(
        tmp_path,
        "a1",
        order=64,
        index=74,
        width=128,
        epochs=60000,
        seed=1,
        selection=_flat_censored(),
    )
    campaign = tmp_path / "campaign.yaml"
    campaign.write_text(
        yaml.safe_dump(
            {
                "cells": [
                    {
                        "phase": "core",
                        "order": 64,
                        "index": 74,
                        "width": 128,
                        "name": "cell-74",
                        "note": "C1 tier-1 pair with (64,80)",
                    },
                    {
                        "phase": "core",
                        "order": 64,
                        "index": 80,
                        "width": 128,
                        "name": "cell-80",
                        "note": "C1 tier-1 pair with (64,74)",
                    },
                ]
            }
        )
    )
    out = tmp_path / "verdict.json"
    rc = grok_verdict.main(
        [
            "--runs-root",
            str(tmp_path),
            "--campaign-config",
            str(campaign),
            "--out",
            str(out),
            "--no-table",
        ]
    )
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["totals"]["n_grokked"] == 1
    assert payload["totals"]["n_censored"] == 1
    assert payload["reconciliation"]["actual_grokked"] == 1
    # The one discovered cell is named from the campaign config.
    names = {c["cell_name"] for c in payload["cells"]}
    assert "cell-74" in names
    # One tier-1 pair found; member (64,80) absent from runs -> missing_member.
    assert payload["tier1_pairs"][0]["status"] == "missing_member"


def test_main_positional_runs_root(tmp_path):
    _write_run(
        tmp_path,
        "a0",
        order=127,
        index=1,
        width=128,
        epochs=60000,
        seed=0,
        selection=_flat_grokked(300),
    )
    campaign = tmp_path / "campaign.yaml"
    campaign.write_text(yaml.safe_dump({"cells": []}))
    out = tmp_path / "verdict.json"
    rc = grok_verdict.main(
        [str(tmp_path), "--campaign-config", str(campaign), "--out", str(out), "--no-table"]
    )
    assert rc == 0
    assert json.loads(out.read_text())["totals"]["n_grokked"] == 1
