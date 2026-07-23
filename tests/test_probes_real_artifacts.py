"""Real-data validation of the mechanism probes on the study's actual targets.

These artifacts (``data/group_artifacts``) and the salvaged v1 checkpoints
(``results-archive``) are Sage/GAP-generated / campaign outputs and are *not*
committed, so every test here skips cleanly when they are absent (CI runs the
fixture corpus only). Where the files exist they pin the decisive properties on
the real groups the paper claims about:

* **I-27** the signed-cyclic coordinate system exists on D32 and D104 and is
  ``UNDEFINED`` on Q32, QD32 and C13:Q8 -- the "must fail on the quaternionic
  member" property, on the real order-32 group.
* **I-26** ``l(G) = 7`` for both C128 and C2^7 while ``d(G) = 1`` (one carry-linked
  direction, triangular carry) vs ``7`` (independent digits, diagonal carry) --
  the C3 headline contrast.

The environment variable ``GROUP_ARTIFACTS_DIR`` is repointed at the real corpus
for the duration of each test and restored afterwards (conftest sets it to the
fixture corpus by default).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parent.parent
_REAL_ARTIFACTS = _REPO / "data" / "group_artifacts"
_ARCHIVE = _REPO / "results-archive"
_SCRIPT_PATH = _REPO / "scripts" / "measure_probes.py"


def _have(order: int, index: int) -> bool:
    return (_REAL_ARTIFACTS / f"smallgroup_{order}_{index}.npz").is_file()


@contextmanager
def _real_artifacts():
    previous = os.environ.get("GROUP_ARTIFACTS_DIR")
    os.environ["GROUP_ARTIFACTS_DIR"] = str(_REAL_ARTIFACTS)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("GROUP_ARTIFACTS_DIR", None)
        else:
            os.environ["GROUP_ARTIFACTS_DIR"] = previous


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("measure_probes", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["measure_probes"] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# I-27 on the real C2 / C4 targets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "order,index,label,expect_signed_cyclic",
    [
        (32, 18, "D32", True),
        (32, 19, "QD32", False),
        (32, 20, "Q32", False),
        (104, 6, "D104", True),
        (104, 4, "C13:Q8", False),
    ],
)
def test_signed_cyclic_on_real_targets(order, index, label, expect_signed_cyclic):
    if not _have(order, index):
        pytest.skip(f"real artifact for {label} ({order},{index}) not present")
    from group_algorithm_interp.groups.catalog import resolve_group
    from group_algorithm_interp.instruments import probes as P

    with _real_artifacts():
        group = resolve_group((order, index))
        coords = P.signed_cyclic_coordinates(group)
    if expect_signed_cyclic:
        assert coords is not None and coords.is_signed_cyclic
        assert coords.radix == order // 2
    else:
        # Q32 (non-split) and QD32 / C13:Q8 (no inversion involution) all fail.
        assert coords is None or not coords.is_signed_cyclic


# ---------------------------------------------------------------------------
# I-26 on the real C3 targets: l(G) constant, d(G) differs 1 vs 7
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "order,index,label,directions,diagonal",
    [
        (128, 1, "C128", 1, False),
        (128, 2328, "C2^7", 7, True),
        (127, 1, "C127", 1, True),
    ],
)
def test_polycyclic_direction_count_on_real_targets(order, index, label, directions, diagonal):
    if not _have(order, index):
        pytest.skip(f"real artifact for {label} ({order},{index}) not present")
    from group_algorithm_interp.groups.catalog import resolve_group
    from group_algorithm_interp.instruments import probes as P

    with _real_artifacts():
        group = resolve_group((order, index))
        pc = P.polycyclic_digits(group)
    assert pc is not None
    if order == 128:
        assert pc.length == 7  # l(G) = 7 for both members
    assert pc.coordinate_directions == directions  # d(G): 1 vs 7
    is_diagonal = bool(np.array_equal(pc.carry, np.eye(pc.length, dtype=bool)))
    assert is_diagonal == diagonal


# ---------------------------------------------------------------------------
# End to end on a salvaged v1 checkpoint (skips when the archive is absent)
# ---------------------------------------------------------------------------


def _find_archive_run(order: int, index: int) -> Path | None:
    if not _ARCHIVE.is_dir():
        return None
    import yaml

    for run_dir in sorted(p for p in _ARCHIVE.iterdir() if p.is_dir()):
        config_path = run_dir / "resolved_config.yaml"
        checkpoint = run_dir / "checkpoints" / "final.pt"
        if not config_path.is_file() or not checkpoint.is_file():
            continue
        config = yaml.safe_load(config_path.read_text())
        group = (config or {}).get("data", {}).get("group", {})
        if group.get("order") == order and group.get("index") == index:
            return run_dir
    return None


@pytest.mark.parametrize(
    "order,index,label,expect_status",
    [(32, 18, "D32", "measured"), (32, 20, "Q32", "UNDEFINED")],
)
def test_probe_run_on_a_salvaged_checkpoint(order, index, label, expect_status):
    if not _have(order, index):
        pytest.skip(f"real artifact for {label} not present")
    run_dir = _find_archive_run(order, index)
    if run_dir is None:
        pytest.skip(f"no salvaged v1 {label} run in results-archive")
    script = _load_script()
    # threshold 0.0 forces final.pt to be selected (these salvaged seeds may be
    # censored below 0.99); the signed-cyclic status is a structural property of
    # the group, not of how well this seed grokked, so it is the same either way.
    with _real_artifacts():
        record = script.probe_run(run_dir, threshold=0.0)
    if record["status"] == "skipped":
        pytest.skip(f"{label} salvaged run has no loadable checkpoint")
    # DECISIVE on real data: signed-cyclic is measured on D32, UNDEFINED on Q32.
    assert record["probes"]["signed_cyclic"]["status"] == expect_status
    assert record["probes"]["power_map"]["metric"] == "adjusted_balanced_accuracy"
