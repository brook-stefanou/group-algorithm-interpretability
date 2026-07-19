"""Test-suite setup.

Belt-and-braces: force W&B into disabled mode at the process level so nothing in
the suite can ever open a network session, even if a test forgets to set
``logging.mode``. Tests also pass ``logging.mode="disabled"`` explicitly, which is
what the trainer hands to ``wandb.init`` -- this env var is the backstop.
"""

import os
import tempfile
from pathlib import Path

import pytest
from support_artifacts import write_test_artifacts

from group_algorithm_interp.seed import set_seed

os.environ["WANDB_MODE"] = "disabled"

# The production corpus is Sage/GAP-generated and intentionally ignored.  Give
# all Python tests a small, generated-at-test-time fixture corpus instead.
#
# Namespacing the directory only by xdist worker id (e.g. "gw0") is not
# enough: worker ids are unique only within one pytest session, so two
# concurrent ``pytest -n auto`` invocations (e.g. ``just gate`` racing a
# manual ``uv run pytest``) can both spawn a "gw0" that resolves to the same
# path and write the same .npz file at once. Use a directory unique per OS
# *process* instead (``tempfile.mkdtemp``), under ``tests/.artifacts``
# (already gitignored) so results stay discoverable for local debugging.
# ``support_artifacts._atomic_savez``'s write-temp-then-``os.replace`` is
# separate defence in depth: it also protects a run killed mid-write (Ctrl-C,
# OOM-kill, CI timeout) from leaving a torn file for a later run to trip over.
_WORKER_ID = os.environ.get("PYTEST_XDIST_WORKER", "master")
_ARTIFACT_BASE = Path(__file__).parent / ".artifacts"
_ARTIFACT_BASE.mkdir(parents=True, exist_ok=True)
_ARTIFACT_DIR = Path(tempfile.mkdtemp(dir=_ARTIFACT_BASE, prefix=f"{_WORKER_ID}-"))
write_test_artifacts(_ARTIFACT_DIR)
os.environ["GROUP_ARTIFACTS_DIR"] = str(_ARTIFACT_DIR)


@pytest.fixture(autouse=True)
def _seed_rngs_before_every_test():
    """Belt-and-braces: reseed Python/NumPy/torch RNGs before every test.

    ``build_model`` (and other helpers) intentionally do not seed -- that is
    ``set_seed``'s job, via the trainer -- so a test that plants weights on a
    fresh model must seed torch itself or inherit whatever RNG state test
    execution order left behind. This fixture fixes that starting state; it
    does not stop a test from taking further random draws within itself.
    """
    set_seed(0, deterministic=False)
