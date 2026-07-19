"""Focused tests for `scripts/preflight.py`.

`preflight.py` answers "can this machine do what I am about to ask it to do?".
This dev host has no CUDA device, no `nvidia-smi`, and no SageMath -- the same
shape of environment as a rented pod, so a wrong answer here is a wrong answer
that costs money there.

Pinned here: the absence of Sage must never fail the `train` path (a pod has
no Sage, by design).

The script is an executable entry point, not part of the installed package, so
it is loaded by file path (mirroring `tests/test_run_batch_command.py`).
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


preflight = _load("preflight")


def _status(checks, name: str) -> str:
    return next(check.status for check in checks if check.name == name)


def _statuses(checks, name: str) -> list[str]:
    return [check.status for check in checks if check.name == name]


# --- the Sage regression -----------------------------------------------------


def test_train_mode_never_checks_sage(capsys):
    """SageMath only regenerates artifacts; a GPU pod's plain `uv sync` has
    none, so its absence must never fail `train`."""
    checks = preflight.run_checks("train", [], None)
    assert not any(check.name == "sage" for check in checks)
    assert all(check.status != "FAIL" or check.name != "sage" for check in checks)


def test_export_mode_checks_sage_and_fails_without_it():
    def _absent(*_args, **_kwargs):
        raise FileNotFoundError("sage")

    check = preflight.check_sage(run=_absent)
    assert check.status == "FAIL"
    assert check.fix is not None and "--extra sage" in check.fix


def test_export_mode_passes_when_sage_runs():
    def _present(*_args, **_kwargs):
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="OK\n", stderr="")

    assert preflight.check_sage(run=_present).status == "OK"


# --- overrides + seeds -------------------------------------------------------


def test_main_accepts_override_and_seeds_flags(capsys):
    """`--seeds` sizes the disk check; `--override` mirrors the campaign
    script's own flag spelling."""
    assert (
        preflight.main(
            ["--override", "device=cpu", "--override", "logging.mode=offline", "--seeds", "0-3"]
        )
        == 0
    )


def test_seeds_parser_expands_ranges_and_rejects_duplicates():
    assert preflight.parse_seeds("0,2,5-7") == [0, 2, 5, 6, 7]
    with pytest.raises(ValueError, match="unique"):
        preflight.parse_seeds("0,0,1")


# --- config ------------------------------------------------------------------


def test_config_check_validates_the_composition_rather_than_a_default_value():
    """A changed snapshot interval is a different experiment, not a broken
    machine."""
    check, config = preflight.check_config(["snapshot.interval=250"])
    assert check.status == "OK"
    assert config.snapshot.interval == 250


def test_config_check_fails_on_an_invalid_override():
    check, config = preflight.check_config(["optim.epochs=not-a-number"])
    assert check.status == "FAIL"
    assert config is None


# --- CUDA --------------------------------------------------------------------


def test_device_cuda_without_a_cuda_device_fails(monkeypatch):
    """The check that stops a 20-seed sweep from silently training on a pod's CPU."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    _, config = preflight.check_config(["device=cuda"])
    checks = preflight.check_cuda(config)
    assert _status(checks, "cuda") == "FAIL"


def test_device_auto_without_a_cuda_device_warns(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    _, config = preflight.check_config([])
    checks = preflight.check_cuda(config)
    assert _status(checks, "cuda") == "WARN"


def test_device_cpu_without_a_cuda_device_is_ok(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    _, config = preflight.check_config(["device=cpu"])
    checks = preflight.check_cuda(config)
    assert _status(checks, "cuda") == "OK"


# --- group artifacts ---------------------------------------------------------


def test_a_missing_artifact_directory_fails(monkeypatch, tmp_path):
    """`data/` is gitignored, so a fresh clone on a pod has no corpus at all."""
    monkeypatch.setenv("GROUP_ARTIFACTS_DIR", str(tmp_path / "absent"))
    _, config = preflight.check_config([])
    check = preflight.check_artifacts(config)
    assert check.status == "FAIL"
    assert check.fix is not None and "GROUP_ARTIFACTS_DIR" in check.fix


def test_a_missing_group_names_the_export_command(monkeypatch, tmp_path):
    monkeypatch.setenv("GROUP_ARTIFACTS_DIR", str(tmp_path))
    _, config = preflight.check_config(["data.group.order=32", "data.group.index=20"])
    check = preflight.check_artifacts(config)
    assert check.status == "FAIL"
    assert check.fix is not None and "--order 32 --index 20" in check.fix


def test_a_present_artifact_loads_and_validates():
    """Loads the file, so one truncated in transit fails here, not at epoch 0."""
    _, config = preflight.check_config([])
    assert preflight.check_artifacts(config).status == "OK"


def test_a_corrupt_artifact_fails_on_load(monkeypatch, tmp_path):
    from group_algorithm_interp.groups.data import artifact_dir, artifact_path

    group_file = artifact_path(8, 1, artifact_dir())
    corrupt = tmp_path / group_file.name
    corrupt.write_bytes(group_file.read_bytes()[: len(group_file.read_bytes()) // 2])
    monkeypatch.setenv("GROUP_ARTIFACTS_DIR", str(tmp_path))
    _, config = preflight.check_config([])
    check = preflight.check_artifacts(config)
    assert check.status == "FAIL"
    assert "validation on load" in check.detail


# --- disk --------------------------------------------------------------------


def test_disk_check_sizes_itself_against_the_seed_count():
    class _Usage:
        free = (20 * preflight.PER_SEED_MIB + preflight.RESERVE_MIB - 1) * 1024**2

    check = preflight.check_disk(list(range(20)), usage=lambda _path: _Usage)
    assert check.status == "FAIL"


def test_disk_check_passes_with_room_for_the_campaign():
    class _Usage:
        free = (20 * preflight.PER_SEED_MIB + preflight.RESERVE_MIB + 1) * 1024**2

    assert preflight.check_disk(list(range(20)), usage=lambda _path: _Usage).status == "OK"


# --- W&B ---------------------------------------------------------------------


def test_online_without_a_key_fails():
    _, config = preflight.check_config(["logging.mode=online"])
    check = preflight.check_wandb(config, key_source=lambda: None, reachable=lambda _host: True)
    assert check.status == "FAIL"
    assert "WANDB_API_KEY" in check.detail


def test_online_with_a_key_but_no_egress_fails():
    _, config = preflight.check_config(["logging.mode=online"])
    check = preflight.check_wandb(
        config, key_source=lambda: "WANDB_API_KEY", reachable=lambda _host: False
    )
    assert check.status == "FAIL"
    assert "unreachable" in check.detail


def test_online_with_a_key_and_egress_passes():
    _, config = preflight.check_config(["logging.mode=online"])
    check = preflight.check_wandb(
        config, key_source=lambda: "WANDB_API_KEY", reachable=lambda _host: True
    )
    assert check.status == "OK"


def test_disabled_logging_warns_rather_than_failing():
    """A paid campaign left unwatched for hours is the real risk here; offline
    dev has no such cost, so this warns instead of failing."""
    _, config = preflight.check_config([])
    check = preflight.check_wandb(config, key_source=lambda: None, reachable=lambda _host: True)
    assert check.status == "WARN"


def test_a_missing_key_is_not_checked_at_all_when_logging_is_offline():
    _, config = preflight.check_config(["logging.mode=offline"])

    def _explode() -> str:
        raise AssertionError("offline mode must not look for a credential")

    assert preflight.check_wandb(config, key_source=_explode).status == "OK"


# --- the plan ----------------------------------------------------------------


def test_the_default_toy_group_warns():
    """Launching a sweep without overriding data.group pays for 20 seeds of C8."""
    _, config = preflight.check_config([])
    assert "WARN" in _statuses(preflight.check_plan(config, list(range(20))), "plan")


def test_a_non_default_group_does_not_warn_about_the_group():
    _, config = preflight.check_config(["data.group.order=6", "data.group.index=1"])
    details = [
        check.detail for check in preflight.check_plan(config, None) if check.status == "WARN"
    ]
    assert not any("repository default" in detail for detail in details)


def test_stop_on_generalize_false_warns_about_the_full_budget():
    _, config = preflight.check_config(["optim.stop_on_generalize=false"])
    warnings = [
        check.detail for check in preflight.check_plan(config, None) if check.status == "WARN"
    ]
    assert any("every seed trains all" in detail for detail in warnings)


def test_stop_on_generalize_true_does_not_warn_about_the_budget():
    _, config = preflight.check_config(["optim.stop_on_generalize=true"])
    warnings = [
        check.detail for check in preflight.check_plan(config, None) if check.status == "WARN"
    ]
    assert not any("every seed trains all" in detail for detail in warnings)


# --- reporting ---------------------------------------------------------------


def test_report_exits_non_zero_on_a_failure_and_reprints_the_fix(capsys):
    checks = [preflight.Check("artifacts", "FAIL", "absent", "copy the corpus")]
    assert preflight.report("train", checks) == 1
    out = capsys.readouterr().out
    assert "not ready to train" in out
    assert "copy the corpus" in out


def test_report_exits_zero_when_only_warnings(capsys):
    checks = [preflight.Check("wandb", "WARN", "disabled", None)]
    assert preflight.report("train", checks) == 0
    assert "1 warning(s)" in capsys.readouterr().out


def test_main_runs_end_to_end_on_this_machine(capsys):
    """No CUDA, no nvidia-smi, no Sage on this host -- must still come out clean."""
    assert preflight.main(["device=cpu", "logging.mode=offline"]) == 0
