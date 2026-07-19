"""Vmapped ensemble entry point: train many seeds of one ``(group, width)`` in a
single batched kernel on one device.

CLI contract (a campaign runner is built against exactly this):

    uv run python scripts/run_batch.py --seeds 0:50
    uv run python scripts/run_batch.py --seeds 0,1,2 experiment=core
    uv run python scripts/run_batch.py --seeds 0:50 data.group.order=32 model.d_model=128

* ``--seeds`` selects the batch. Accepts a comma-separated mix of half-open
  ranges (``a:b`` -> ``a, a+1, ..., b-1``) and individual integers, e.g.
  ``0:50`` (seeds 0..49), ``0,1,2``, or ``0:10,20,30:32``. Order is preserved
  and duplicates are removed. The batch size ``K`` is ``len(seeds)`` -- the
  caller picks ``K`` (and thus peak memory) by choosing how many seeds to hand
  one process.
* Every remaining argument is a Hydra override, composed against ``configs/``
  exactly as ``scripts/run.py`` composes it -- so ``experiment=core``,
  ``model.d_model=128``, ``data.group.order=32`` all work identically.
* ``--runs-root PATH`` (optional, default ``runs``) relocates the per-seed run
  directories -- used by tests; campaigns normally leave it at the default.
* ``data.split_seed`` must be pinned (its default is). All ``K`` models share one
  split; the trainer refuses ``data.split_seed=null``.

Memory-aware chunking: per-model peak memory scales roughly with
``|G|^2 * width``, so a large ``(group, width)`` with many seeds can exceed one
card. ``run_batch`` therefore does NOT necessarily stack all requested seeds into
one ensemble -- it splits them into consecutive, memory-sized sub-batches
("chunks") and trains each to completion before the next. On CUDA the chunk size
is measured empirically: before building the full ensemble a K=1 trial for this
exact ``(group, width)`` measures the true per-model peak, and the chunk size is
``floor((free_memory * safety) / per_model_cost)``, clamped to ``[1, K]``. Chunks
are packed to the maximum (50 seeds at max 30 -> 30 + 20, not a balanced split),
minimising the chunk count. The decision (per-model MiB, free MiB, chunk size,
chunk count) is logged loudly. On a non-CUDA device there is one chunk of every
seed unless an override forces otherwise. Two flags tune it:

* ``--max-models-per-batch N`` -- a hard ceiling that skips estimation entirely
  and forces chunks of at most ``N`` models.
* ``--memory-safety-fraction F`` -- the headroom fraction (default ``0.8``).

Chunking is internal to ``run_batch`` and does not change the CLI contract the
campaign runner depends on: seeds still split in order, each seed still gets one
single-run-schema run directory, and the exit code is still ``0`` iff every seed
completed -- so ``scripts/run_campaign.py`` needs no change to benefit from it.

Provenance and W&B: this path writes one run directory per seed -- manifest,
resolved_config, ``run.log`` eval history, and checkpoints -- with the single-run
schema and a distinct run id per seed, so ``scripts/sync_runs.py`` and the
analysis pipeline read a batch-trained seed exactly as a single run. It records
to **manifests and eval-history only**; per-model W&B runs are intentionally not
created here (they are optional and default-off), and ``sync_runs.py`` uploads
the run artifacts later.

Exit code: ``0`` iff every seed's manifest reached ``completed``; ``1``
otherwise. A SIGTERM / Ctrl-C finalises every unfinished manifest to ``aborted``
before exiting.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# See scripts/run.py: cuBLAS reads this when it creates its handle, and a
# deterministic run needs it set before torch touches CUDA. Inert unless
# determinism is opted in; setting it is what makes that opt-in work.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from collections.abc import Sequence  # noqa: E402

from hydra import compose, initialize_config_dir  # noqa: E402

from group_algorithm_interp.config import validate_config  # noqa: E402
from group_algorithm_interp.dotenv import load_dotenv  # noqa: E402
from group_algorithm_interp.training.ensemble import BatchEnsembleTrainer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"

load_dotenv()


def parse_seeds(spec: str) -> list[int]:
    """Parse a ``--seeds`` spec into an ordered, de-duplicated list of seeds.

    Accepts a comma-separated mix of half-open ranges ``a:b`` (yields
    ``a..b-1``) and single integers, e.g. ``0:50``, ``0,1,2``, ``0:10,20``.
    Order is the order of first appearance; duplicates are dropped.
    """
    seeds: list[int] = []
    seen: set[int] = set()

    def _add(value: int) -> None:
        if value not in seen:
            seen.add(value)
            seeds.append(value)

    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            start_str, end_str = token.split(":", 1)
            start, end = int(start_str), int(end_str)
            if end < start:
                raise ValueError(f"empty/reversed seed range {token!r}: need a:b with b >= a")
            for value in range(start, end):
                _add(value)
        else:
            _add(int(token))
    if not seeds:
        raise ValueError(f"--seeds {spec!r} selected no seeds")
    return seeds


def _split_argv(argv: Sequence[str]) -> tuple[str, str, int | None, float, list[str]]:
    """Pull ``--seeds`` and the optional ``--runs-root`` /
    ``--max-models-per-batch`` / ``--memory-safety-fraction`` values out of
    ``argv``; everything else is a Hydra override. Every flag supports both
    ``--flag VALUE`` and ``--flag=VALUE``.

    Returns ``(seeds_spec, runs_root, max_models_per_batch, safety, overrides)``."""
    seeds_spec: str | None = None
    runs_root = "runs"
    max_models_per_batch: int | None = None
    safety = 0.8
    overrides: list[str] = []
    i = 0
    args = list(argv)

    def _value(flag: str, index: int) -> str:
        if index + 1 >= len(args):
            raise ValueError(f"{flag} requires a value")
        return args[index + 1]

    while i < len(args):
        arg = args[i]
        if arg == "--seeds":
            seeds_spec = _value(arg, i)
            i += 2
            continue
        if arg.startswith("--seeds="):
            seeds_spec = arg[len("--seeds=") :]
            i += 1
            continue
        if arg == "--runs-root":
            runs_root = _value(arg, i)
            i += 2
            continue
        if arg.startswith("--runs-root="):
            runs_root = arg[len("--runs-root=") :]
            i += 1
            continue
        if arg == "--max-models-per-batch":
            max_models_per_batch = int(_value(arg, i))
            i += 2
            continue
        if arg.startswith("--max-models-per-batch="):
            max_models_per_batch = int(arg[len("--max-models-per-batch=") :])
            i += 1
            continue
        if arg == "--memory-safety-fraction":
            safety = float(_value(arg, i))
            i += 2
            continue
        if arg.startswith("--memory-safety-fraction="):
            safety = float(arg[len("--memory-safety-fraction=") :])
            i += 1
            continue
        overrides.append(arg)
        i += 1
    if seeds_spec is None:
        raise ValueError("--seeds is required (e.g. --seeds 0:50 or --seeds 0,1,2)")
    return seeds_spec, runs_root, max_models_per_batch, safety, overrides


def main(argv: Sequence[str] | None = None) -> int:
    seeds_spec, runs_root, max_models_per_batch, safety, overrides = _split_argv(
        sys.argv[1:] if argv is None else argv
    )
    seeds = parse_seeds(seeds_spec)
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        composed = compose(config_name="config", overrides=overrides)
    config = validate_config(composed)

    trainer = BatchEnsembleTrainer(
        config,
        seeds,
        runs_root=runs_root,
        max_models_per_batch=max_models_per_batch,
        memory_safety_fraction=safety,
    )
    statuses = trainer.run()
    completed = sum(1 for status in statuses.values() if status == "completed")
    print(f"batch: {completed}/{len(seeds)} seed(s) completed")
    return 0 if completed == len(seeds) else 1


if __name__ == "__main__":
    raise SystemExit(main())
