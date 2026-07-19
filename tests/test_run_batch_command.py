"""Focused tests for `scripts/run_batch.py`'s CLI contract.

The campaign runner is built against exactly this contract: ``--seeds`` takes a
comma-separated mix of half-open ranges (``a:b``) and integers, every other
argument is a Hydra override composed against ``configs/`` like ``scripts/run.py``,
and the exit code is 0 iff every seed's manifest reached ``completed``.

The script is an executable entry point, not part of the installed package, so
it is loaded by file path (mirroring ``tests/test_run_campaign.py``).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "run_batch.py"


def _load_run_batch() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_batch", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_batch = _load_run_batch()


class TestParseSeeds:
    def test_half_open_range(self):
        assert run_batch.parse_seeds("0:5") == [0, 1, 2, 3, 4]

    def test_comma_list(self):
        assert run_batch.parse_seeds("0,1,2") == [0, 1, 2]

    def test_mixed_ranges_and_singles_preserve_order_and_dedupe(self):
        assert run_batch.parse_seeds("0:3,2,10,8:10") == [0, 1, 2, 10, 8, 9]

    def test_empty_range_rejected(self):
        with pytest.raises(ValueError, match="reversed"):
            run_batch.parse_seeds("5:0")

    def test_no_seeds_rejected(self):
        with pytest.raises(ValueError, match="selected no seeds"):
            run_batch.parse_seeds("0:0")

    def test_garbage_rejected(self):
        with pytest.raises(ValueError):
            run_batch.parse_seeds("a-b")


class TestSplitArgv:
    def test_seeds_required(self):
        with pytest.raises(ValueError, match="--seeds is required"):
            run_batch._split_argv(["optim.epochs=2"])

    def test_flags_and_overrides_separate(self):
        seeds, root, max_models, safety, overrides = run_batch._split_argv(
            ["--seeds", "0:3", "--runs-root", "/tmp/x", "experiment=core", "model.d_model=128"]
        )
        assert seeds == "0:3"
        assert root == "/tmp/x"
        assert max_models is None  # unset -> estimation decides
        assert safety == 0.8  # default headroom
        assert overrides == ["experiment=core", "model.d_model=128"]

    def test_equals_form(self):
        seeds, root, max_models, safety, overrides = run_batch._split_argv(
            ["--seeds=0,1", "--runs-root=/tmp/y"]
        )
        assert seeds == "0,1"
        assert root == "/tmp/y"
        assert max_models is None
        assert safety == 0.8
        assert overrides == []

    def test_chunking_flags_space_form(self):
        seeds, _root, max_models, safety, overrides = run_batch._split_argv(
            ["--seeds", "0:50", "--max-models-per-batch", "30", "--memory-safety-fraction", "0.7"]
        )
        assert seeds == "0:50"
        assert max_models == 30
        assert safety == 0.7
        assert overrides == []

    def test_chunking_flags_equals_form(self):
        seeds, _root, max_models, safety, overrides = run_batch._split_argv(
            ["--seeds=0:50", "--max-models-per-batch=17", "--memory-safety-fraction=0.9"]
        )
        assert seeds == "0:50"
        assert max_models == 17
        assert safety == 0.9
        assert overrides == []

    def test_chunking_flags_are_not_hydra_overrides(self):
        """Must be stripped from the Hydra override list, or they'd be composed
        against configs/ and fail."""
        _seeds, _root, max_models, safety, overrides = run_batch._split_argv(
            ["--seeds", "0:4", "--max-models-per-batch", "2", "model.d_model=8"]
        )
        assert max_models == 2
        assert safety == 0.8
        assert overrides == ["model.d_model=8"]


def test_main_end_to_end_exit_zero_iff_all_completed(tmp_path):
    """The whole contract in one shot: Hydra composition against configs/, a
    K=2 batch trains, each seed's manifest reaches ``completed``."""
    code = run_batch.main(
        [
            "--seeds",
            "0,1",
            "--runs-root",
            str(tmp_path),
            "model.d_model=8",
            "model.d_mlp=16",
            "model.n_heads=1",
            "data.group=C4",
            "data.train_frac=0.5",
            "optim.epochs=2",
            "optim.log_every=1",
            "snapshot.enabled=false",
            "logging.mode=disabled",
        ]
    )
    assert code == 0

    run_dirs = sorted(p for p in tmp_path.iterdir() if (p / "manifest.yaml").is_file())
    assert len(run_dirs) == 2
    seeds_on_record = set()
    for run_dir in run_dirs:
        manifest = yaml.safe_load((run_dir / "manifest.yaml").read_text())
        assert manifest["status"] == "completed"
        resolved = yaml.safe_load((run_dir / "resolved_config.yaml").read_text())
        seeds_on_record.add(resolved["seed"])
    assert seeds_on_record == {0, 1}
