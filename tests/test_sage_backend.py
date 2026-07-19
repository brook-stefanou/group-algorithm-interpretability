"""Live regression panel for the Sage/GAP exporter.

Runs the exporter *as a subprocess, right now*, and so needs Sage present: it
is SKIPPED wherever Sage is not installed, which includes CI. Install with
``uv sync --extra sage`` (pip-installable passagemath, which also puts a
``sage`` binary on PATH).

``tests/data/group_artifacts`` holds golden artifacts produced by the real
exporter and committed to the repository; ``tests/test_representation_ground_truth.py``
runs its full ground-truth panel against them with no Sage installed, and
``tests/test_golden_sage_artifacts.py`` cross-checks them against the
independent hand-built fixtures. What only this file still adds: it catches an
exporter regression *at the moment it is introduced*, on a group whose
artifact nobody has regenerated yet -- the golden corpus only catches one once
someone re-exports.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from group_algorithm_interp.groups.data import load_group


def test_missing_artifact_explains_sage_setup(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"sage -python scripts/export_group.py"):
        load_group(8, 3, tmp_path)


@pytest.mark.skipif(
    shutil.which("sage") is None,
    reason=(
        "SageMath is not installed, so the exporter cannot be RE-RUN here "
        "(install: uv sync --extra sage). The exporter's output is still "
        "under test in this environment: tests/test_representation_ground_truth.py "
        "runs the full ground-truth panel against the committed Sage-produced "
        "artifacts in tests/data/group_artifacts, and "
        "tests/test_golden_sage_artifacts.py cross-checks them against the "
        "hand-built fixtures. Both are Sage-free."
    ),
)
@pytest.mark.parametrize(
    ("order", "index", "dimensions"),
    [(4, 1, [1, 1, 1, 1]), (6, 1, [1, 1, 2]), (8, 4, [1, 1, 1, 1, 2])],
)
def test_sage_export_regression_panel(
    tmp_path: Path, order: int, index: int, dimensions: list[int]
) -> None:
    artifact = tmp_path / f"smallgroup_{order}_{index}.npz"
    subprocess.run(
        [
            "sage",
            "-python",
            "scripts/export_group.py",
            "--order",
            str(order),
            "--index",
            str(index),
            "--output",
            str(artifact),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    group = load_group(order, index, tmp_path)
    assert group.canonical_id == (order, index)
    assert group.cayley_table.shape == (order, order)
    assert sorted(irrep.dimension for irrep in group.irreps) == dimensions
    assert len(group.conjugacy_classes) == group.character_table.shape[0]
    for irrep in group.irreps:
        assert np.allclose(np.trace(irrep.matrices, axis1=1, axis2=2), irrep.character)
        for left in group.elements:
            for right in group.elements:
                assert np.allclose(
                    irrep.matrices[left] @ irrep.matrices[right],
                    irrep.matrices[group.cayley_table[left, right]],
                )
    # "block_rank == trace(projector)" and "ranks sum to order" alone would not
    # have caught the historical isotypic-dimension bug (IsotypicBlock.dimension
    # conflated irrep_degree and block_rank for every irrep of degree >= 2): a
    # self-consistent-but-wrong block_rank could still satisfy both. The check
    # that would have caught it is block_rank against sum_i d_i**2 over the
    # block's irreps, independently derived from irrep.dimension.
    for block in group.isotypic_blocks:
        assert np.allclose(block.projector @ block.projector, block.projector, atol=1e-9)
        assert np.isclose(block.block_rank, np.trace(block.projector))
        expected_rank = sum(group.irreps[i].dimension ** 2 for i in block.irrep_indices)
        assert block.block_rank == expected_rank
    assert sum(block.block_rank for block in group.isotypic_blocks) == order
    assert group.provenance["backend"] == "SageMath/libgap"
