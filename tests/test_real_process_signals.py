"""Real-process signal test: a run that is SIGTERMed must finalise its manifest.

The rest of the suite can't see the bug this guards: every other trainer
signal test drives a ``_FakeProcess`` or a monkeypatched ``Popen``, and a
double has no real ``uv`` parent, so it delivers exactly one signal. Reality
delivers two.

A campaign runner's drain SIGTERMs the child's whole **process group**,
which contains both ``uv`` and the trainer underneath it. ``uv`` then also
forwards its own SIGTERM down to that same trainer, so the trainer receives
SIGTERM **twice**, milliseconds apart. If ``experiment._raise_aborted`` is
still armed when the second one lands, it raises ``RunAborted`` from inside
``finalize_manifest``'s read-modify-write, the write is abandoned, and the
manifest is stuck at ``status: running`` forever -- indistinguishable from a
run still training, the exact state the SIGTERM handler exists to prevent.
Measured before the fix: the manifest was lost in roughly half of all
``killpg`` trials.

So this spawns a REAL ``uv run python scripts/run.py``, lets it reach training,
and ``killpg``s it exactly as a campaign runner's drain does. Delete the
``SIG_IGN`` self-disarm from ``_raise_aborted`` and this goes red.

Opt-in (it costs seconds, spawns real processes, and needs ``uv`` on PATH), using
the same env-var convention as the full-determinism test in
``tests/test_smoke_train.py``::

    RUN_REAL_PROCESS_SIGNAL_TEST=1 uv run pytest tests/test_real_process_signals.py
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from contextlib import suppress
from pathlib import Path

import pytest
import yaml

_OPT_IN_ENV_VAR = "RUN_REAL_PROCESS_SIGNAL_TEST"

ROOT = Path(__file__).resolve().parent.parent
GOLDEN_ARTIFACTS = ROOT / "tests" / "data" / "group_artifacts"

# The double delivery is a race: the second SIGTERM has to land inside the
# finalise window. One trial is not enough to see a ~50% failure -- repeat until
# missing it is unlikely (0.5**5 ~= 3%).
TRIALS = 5

# Long enough that the child is certainly still training when the signal lands
# (allow_expensive: this is over eval.max_steps_warn).
CHILD_OVERRIDES = [
    "experiment=debug",
    "device=cpu",
    "optim.epochs=200000",
    "allow_expensive=true",
    "optim.print_every=5000",
    "snapshot.enabled=false",
]

pytestmark = pytest.mark.skipif(
    os.environ.get(_OPT_IN_ENV_VAR) != "1" or shutil.which("uv") is None,
    reason=(
        "Opt-in: spawns real `uv run` subprocesses and signals real process "
        f"groups, which is too slow and too environment-dependent for the gate. "
        f"Set {_OPT_IN_ENV_VAR}=1 (with `uv` on PATH) to run it."
    ),
)


def _wait_for(predicate, timeout: float, what: str):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.05)
    raise AssertionError(f"timed out after {timeout}s waiting for {what}")


def _training_run_dir(runs_root: Path) -> Path | None:
    """The child's run directory, once it is genuinely past setup and training."""
    if not runs_root.is_dir():
        return None
    for candidate in runs_root.iterdir():
        log = candidate / "run.log"
        if log.exists() and "epoch 0" in log.read_text():
            return candidate
    return None


def _run_one_trial(tmp_path: Path) -> dict:
    """Launch a real child, let it train, SIGTERM its process group, return its manifest."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    command = [
        "uv",
        "run",
        "--project",
        str(ROOT),
        "python",
        str(ROOT / "scripts" / "run.py"),
        *CHILD_OVERRIDES,
    ]
    environment = {
        **os.environ,
        "WANDB_MODE": "disabled",
        "GROUP_ARTIFACTS_DIR": str(GOLDEN_ARTIFACTS),
    }
    # cwd=tmp_path so the child's runs/ and Hydra outputs/ land in the tmp dir,
    # not in the repository. start_new_session so it leads its own process group,
    # exactly as a campaign runner starts every seed.
    process = subprocess.Popen(
        command,
        cwd=tmp_path,
        env=environment,
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        run_dir = _wait_for(
            lambda: _training_run_dir(tmp_path / "runs"), 180.0, "the child to start training"
        )
        manifest_path = run_dir / "manifest.yaml"
        assert yaml.safe_load(manifest_path.read_text())["status"] == "running"

        # Exactly what a campaign runner's drain does: signal the GROUP, which
        # reaches uv and the trainer under it -- and uv forwards its copy down as
        # well, so the trainer is signalled twice.
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        exit_code = process.wait(timeout=120)

        # The runner reads the manifest as soon as its handle exits, so the
        # manifest must already be terminal at that point -- no sleep here.
        manifest = yaml.safe_load(manifest_path.read_text())
        manifest["_exit_code"] = exit_code
        return manifest
    finally:
        if process.poll() is None:  # never leave a trainer behind, even on failure
            with suppress(OSError):
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            process.wait(timeout=30)


def test_group_sigterm_finalises_the_manifest_every_time(tmp_path: Path) -> None:
    """A killpg'd run reaches `aborted` -- never `running` -- on every trial.

    Goes red if ``_raise_aborted`` stops disarming itself: the second SIGTERM
    (forwarded by uv) then re-enters the handler inside ``finalize_manifest`` and
    strands the manifest at ``status: running``.
    """
    statuses = []
    for trial in range(TRIALS):
        manifest = _run_one_trial(tmp_path / f"trial{trial}")
        statuses.append(manifest["status"])

        assert manifest["status"] == "aborted", (
            f"trial {trial}: manifest is {manifest['status']!r}, not 'aborted'. A "
            "SIGTERM to the process group is delivered twice (directly, and again "
            "forwarded by uv); the second delivery re-entered the abort handler "
            "inside finalize_manifest and abandoned the write. See "
            "experiment._raise_aborted's SIG_IGN self-disarm. "
            f"statuses so far: {statuses}"
        )
        assert manifest["completed_at"] is not None, f"trial {trial}: completed_at was not set"
        assert "SIGTERM" in str(manifest["error"]), (
            f"trial {trial}: the manifest error does not name the signal: {manifest['error']!r}"
        )
        assert manifest["_exit_code"] != 0, f"trial {trial}: a terminated run exited 0"
