"""Run entry point.

Hydra composes the config from ``configs/`` (plus any CLI overrides); we validate
it with Pydantic and hand the typed object to the experiment. Examples::

    uv run python scripts/run.py
    uv run python scripts/run.py optim.lr=1e-3 optim.epochs=100 model.d_model=128
    uv run python scripts/run.py logging.mode=offline
    uv run python scripts/run.py -m seed=0,1,2 optim.lr=1e-3,3e-4   # multirun

This entry point runs the project's full-batch group-multiplication experiment.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# cuBLAS reads CUBLAS_WORKSPACE_CONFIG when it creates its handle (the first CUDA
# matmul); torch.use_deterministic_algorithms(True) refuses to run a cuBLAS op
# without it, so under `deterministic: true` the model's first einsum would raise
# on every GPU seed. Determinism is opt-in and off by default (CUDA runs are not
# bit-reproducible; see docs/reproducibility.md), so this variable is normally
# inert -- but it must be set before torch is imported for that opt-in to work.
# Set it here and at package import (group_algorithm_interp.seed) so both `python
# scripts/run.py` and a bare `import group_algorithm_interp` are covered.
# setdefault: an operator's explicit value wins.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# Make `src/` importable without relying on the editable-install .pth: on macOS
# that .pth can get flagged UF_HIDDEN, and Python 3.12+ silently skips hidden
# .pth files, so a freshly synced venv can fail to import the package. This line
# keeps `uv run python scripts/run.py` working regardless.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import hydra  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from group_algorithm_interp.config import validate_config  # noqa: E402
from group_algorithm_interp.dotenv import load_dotenv  # noqa: E402
from group_algorithm_interp.experiment import GroupGeneralizationExperiment  # noqa: E402

# Pick up WANDB_* / keys from a local .env (uv run does not load it automatically).
load_dotenv()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    config = validate_config(cfg)  # Hydra composes; Pydantic validates.
    experiment = GroupGeneralizationExperiment(config)
    summary = experiment.execute()
    log = logging.getLogger("group_algorithm_interp")
    log.info("final: %s", {k: round(v, 4) for k, v in summary.items()})
    log.info("run dir: %s", experiment.run_dir)


if __name__ == "__main__":
    main()
