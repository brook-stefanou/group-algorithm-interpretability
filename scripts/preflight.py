#!/usr/bin/env python3
"""Check that this machine can do the thing it is about to be asked to do.

Two modes, because the two workflows need different machines:

``train`` (the default)
    What a GPU pod needs: CUDA visible to torch, the group artifact the config
    names present on disk and loadable, room under ``runs/`` for the
    checkpoints, and a W&B key when the config logs online. SageMath is not
    checked: training reads exported ``.npz`` artifacts and never imports Sage.

``export`` (a workstation regenerating artifacts)
    What ``scripts/export_group.py`` needs: a working Sage/GAP.

The train checks are run against the config that is actually about to run, so
pass the same Hydra overrides (``key=value`` or ``--override key=value``) the
campaign will use -- ``scripts/run_campaign.py``/``scripts/run_batch.py`` take
those same bare overrides -- so preflight and the campaign check the same
config::

    uv run python scripts/preflight.py
    uv run python scripts/preflight.py data.group.order=32 data.group.index=20
    uv run python scripts/preflight.py --seeds 0-49 \
      --override data.group.order=32 --override data.group.index=18 \
      --override logging.mode=online
    uv run python scripts/preflight.py --mode export

Exit codes: ``0`` no failures (warnings are printed but do not fail), ``1`` at
least one check failed.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import socket
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def parse_seeds(value: str) -> list[int]:
    """Parse ``0,2,5-7`` into explicit, unique seeds -- used only to size the
    disk check against a campaign's seed count."""
    seeds: list[int] = []
    for token in (part.strip() for part in value.split(",")):
        if not token:
            continue
        bounds = token.split("-", maxsplit=1)
        try:
            if len(bounds) == 1:
                seeds.append(int(bounds[0]))
            else:
                start, end = (int(bound) for bound in bounds)
                if end < start:
                    raise ValueError
                seeds.extend(range(start, end + 1))
        except ValueError as exc:
            raise ValueError(f"invalid seed expression {token!r}") from exc
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be non-empty and unique")
    return seeds


# Measured on a 10,000-epoch run: one seed's runs/<run_id>/ (manifest, resolved
# config, log, trajectory snapshots) is dominated by the 112 snapshots the
# default policy writes. At the default `model.d_model: 128` the model holds
# ~134k parameters, a snapshot is ~527 KiB, and the run lands at ~58 MiB. The
# group's order contributes only an `order x d_model` embedding and is noise at
# this scale; `d_model` is what moves this number (it was ~22 MiB at d_model 64).
# Size the disk check just above the measurement.
PER_SEED_MIB = 60

# Free space to keep beyond what the seeds themselves need: the uv cache, the
# torch wheel, and the pod's own logs all live on the same volume.
RESERVE_MIB = 2048

WANDB_HOST = "api.wandb.ai"

Status = Literal["OK", "WARN", "FAIL"]
Mode = Literal["train", "export"]


@dataclass(frozen=True)
class Check:
    """One question about this machine, and the answer.

    ``fix`` is the command or environment change that clears a FAIL. It is
    reprinted in the summary, so an operator who scrolled past the check still
    sees what to do.
    """

    name: str
    status: Status
    detail: str
    fix: str | None = None


def compose_config(overrides: Sequence[str]) -> Any:
    """Compose and validate exactly what ``scripts/run.py`` would build."""
    from hydra import compose, initialize_config_dir

    from group_algorithm_interp.config import validate_config

    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        composed = compose(config_name="config", overrides=list(overrides))
    return validate_config(composed)


def check_python() -> Check:
    """torch, the package, and the rest of the runtime import."""
    missing = [
        name
        for name in ("numpy", "torch", "yaml", "hydra", "omegaconf", "pydantic", "wandb")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        return Check(
            "python",
            "FAIL",
            f"missing: {', '.join(missing)}",
            "uv sync",
        )
    if importlib.util.find_spec("group_algorithm_interp") is None:
        return Check("python", "FAIL", "group_algorithm_interp is not importable", "uv sync")
    import torch

    version = ".".join(str(part) for part in sys.version_info[:3])
    return Check("python", "OK", f"python {version}, torch {torch.__version__}, package importable")


def check_config(overrides: Sequence[str]) -> tuple[Check, Any]:
    """The config composes and passes Pydantic validation.

    Asserts nothing about the *value* of any default -- an operator who changes a
    snapshot interval or an epoch budget has changed the experiment, not broken
    the environment.
    """
    try:
        config = compose_config(overrides)
    except Exception as exc:
        return (
            Check(
                "config",
                "FAIL",
                f"{type(exc).__name__}: {exc}",
                "fix the overrides, or drop them and re-run",
            ),
            None,
        )
    shown = " ".join(overrides) if overrides else "no overrides (repository defaults)"
    return Check("config", "OK", f"composes and validates: {shown}"), config


def check_cuda(config: Any) -> list[Check]:
    """torch sees the device the config asks to train on.

    A campaign shard runs on the device its config selects (masked to one GPU
    with ``CUDA_VISIBLE_DEVICES``), so what matters here is that ``torch.cuda``
    actually sees a device when the config wants one -- the check that stops a
    ``device=cuda`` sweep from silently falling back to a pod's CPU.
    """
    import torch

    available = torch.cuda.is_available()
    wanted = config.device  # "auto" | "cpu" | "cuda" | "mps"
    if available:
        devices = [
            f"{index}: {torch.cuda.get_device_name(index)} "
            f"({torch.cuda.get_device_properties(index).total_memory // 1024**2} MiB)"
            for index in range(torch.cuda.device_count())
        ]
        return [
            Check("cuda", "OK", f"{torch.cuda.device_count()} device(s) — " + "; ".join(devices))
        ]
    if wanted == "cuda":
        return [
            Check(
                "cuda",
                "FAIL",
                "config sets device=cuda but torch.cuda.is_available() is False",
                "use a CUDA host, or check the driver with `nvidia-smi`",
            )
        ]
    if wanted == "auto":
        return [
            Check(
                "cuda",
                "WARN",
                "no CUDA device; device=auto will resolve to CPU and train there",
                "on a GPU host this is a misconfiguration — every seed would run on CPU",
            )
        ]
    return [Check("cuda", "OK", f"no CUDA device; config sets device={wanted}")]


def check_artifacts(config: Any) -> Check:
    """The group the config names is on this disk and loads.

    The corpus is Sage-generated into a gitignored ``data/``, so a fresh clone on
    a pod has none of it: this is the check that catches an artifact directory
    that was never copied across. Loading (not just stat-ing) the artifact also
    re-validates its group axioms and isotypic blocks, which catches a file
    truncated in transit.
    """
    from group_algorithm_interp.groups.data import artifact_dir, artifact_path, load_group

    group = config.data.group
    directory = artifact_dir()
    source = (
        f"GROUP_ARTIFACTS_DIR={os.environ['GROUP_ARTIFACTS_DIR']}"
        if os.environ.get("GROUP_ARTIFACTS_DIR")
        else f"default {directory} (GROUP_ARTIFACTS_DIR unset)"
    )
    path = artifact_path(group.order, group.index, directory)
    if not directory.is_dir():
        return Check(
            "artifacts",
            "FAIL",
            f"artifact directory does not exist: {source}",
            "copy the corpus to this host and point GROUP_ARTIFACTS_DIR at it",
        )
    if not path.exists():
        present = sorted(entry.name for entry in directory.glob("smallgroup_*.npz"))
        return Check(
            "artifacts",
            "FAIL",
            f"{group.canonical_name} needs {path.name}, absent from {source}; "
            f"present: {', '.join(present) if present else 'nothing'}",
            f"export it: uv run python scripts/export_group.py "
            f"--order {group.order} --index {group.index}",
        )
    try:
        load_group(group.order, group.index, directory)
    except Exception as exc:
        return Check(
            "artifacts",
            "FAIL",
            f"{path.name} failed validation on load: {type(exc).__name__}: {exc}",
            "re-export or re-copy the artifact; it is corrupt or stale",
        )
    return Check("artifacts", "OK", f"{group.canonical_name} loads and validates from {source}")


def check_disk(
    seeds: Sequence[int] | None, usage: Callable[[Path], Any] = shutil.disk_usage
) -> Check:
    """Enough room under ``runs/`` for the checkpoints this campaign will write."""
    free_mib = int(usage(ROOT).free) // 1024**2
    if seeds is None:
        if free_mib < RESERVE_MIB:
            return Check(
                "disk",
                "FAIL",
                f"{free_mib} MiB free at {ROOT}, below the {RESERVE_MIB} MiB floor",
                "free space, or run from a larger volume",
            )
        return Check(
            "disk",
            "OK",
            f"{free_mib} MiB free at {ROOT} (pass --seeds to size this against a campaign)",
        )
    needed = len(seeds) * PER_SEED_MIB + RESERVE_MIB
    detail = (
        f"{free_mib} MiB free at {ROOT}; {len(seeds)} seed(s) need about "
        f"{len(seeds) * PER_SEED_MIB} MiB plus {RESERVE_MIB} MiB reserve"
    )
    if free_mib < needed:
        return Check("disk", "FAIL", detail, "free space, or run from a larger volume")
    return Check("disk", "OK", detail)


def _wandb_key_source() -> str | None:
    """Where a W&B credential would come from, or ``None`` if there is none.

    ``scripts/run.py`` calls ``load_dotenv()``, so a ``.env`` in the repository
    root is a real credential source -- and one a pod does *not* get from a
    ``git clone``, because ``.env`` is gitignored. Check it the same way the
    trainer will, rather than only reading the current environment.
    """
    from group_algorithm_interp.dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    if os.environ.get("WANDB_API_KEY"):
        return "WANDB_API_KEY"
    try:
        import netrc

        hosts = netrc.netrc().hosts
    except Exception:
        return None
    return "~/.netrc" if WANDB_HOST in hosts else None


def _reachable(host: str, port: int = 443, timeout: float = 5.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def check_wandb(
    config: Any,
    key_source: Callable[[], str | None] = _wandb_key_source,
    reachable: Callable[[str], bool] = _reachable,
) -> Check:
    """A pod that logs nowhere is a pod that bills by the hour and shows nothing.

    ``logging.mode=online`` fails the run at ``wandb.init`` without a credential
    or without egress, so both are hard failures here. ``disabled`` is not an
    error -- it is the default, and it is right for the offline gate -- but on a
    long GPU campaign it means no live metrics, so it warns.
    """
    mode = config.logging.mode
    if mode == "disabled":
        return Check(
            "wandb",
            "WARN",
            "logging.mode=disabled: no metrics leave this machine and nothing is "
            "watchable while the campaign runs",
            "pass logging.mode=online (with WANDB_API_KEY) to watch a paid run",
        )
    if mode == "offline":
        return Check(
            "wandb", "OK", "logging.mode=offline: metrics buffer to ./wandb for `wandb sync`"
        )
    source = key_source()
    if source is None:
        return Check(
            "wandb",
            "FAIL",
            "logging.mode=online but no credential: WANDB_API_KEY is unset and ~/.netrc "
            "has no api.wandb.ai entry. .env is gitignored, so a clone does not bring one",
            "export WANDB_API_KEY=... in the pod environment (or run `wandb login`)",
        )
    if not reachable(WANDB_HOST):
        return Check(
            "wandb",
            "FAIL",
            f"logging.mode=online and a credential is present ({source}), but {WANDB_HOST}:443 "
            "is unreachable; wandb.init will fail and take the run with it",
            "open egress from this host, or use logging.mode=offline and `wandb sync` later",
        )
    return Check(
        "wandb", "OK", f"logging.mode=online, credential from {source}, {WANDB_HOST} reachable"
    )


def check_plan(config: Any, seeds: Sequence[int] | None) -> list[Check]:
    """The cost-relevant knobs, and the two that quietly waste a pod.

    The repository default group is a toy; the default budget runs every epoch
    even after a seed generalises. Both are legitimate settings and neither is
    an environment fault, so both warn rather than fail -- but they warn *before*
    the invoice, which is the point of running this.
    """
    checks: list[Check] = []
    group = config.data.group
    epochs = config.optim.epochs
    count = len(seeds) if seeds is not None else 1
    checks.append(
        Check(
            "plan",
            "OK",
            f"{count} seed(s) x {epochs} epochs on {group.canonical_name}, "
            f"device={config.device}, stop_on_generalize={config.optim.stop_on_generalize}",
        )
    )
    default_group = compose_config([]).data.group
    if group.canonical_id == default_group.canonical_id:
        checks.append(
            Check(
                "plan",
                "WARN",
                f"data.group is the repository default {group.canonical_name}, a toy group "
                "for exercising the plumbing",
                "pass --override data.group.order=... --override data.group.index=... "
                "to the campaign (and to this script)",
            )
        )
    if not config.optim.stop_on_generalize:
        checks.append(
            Check(
                "plan",
                "WARN",
                f"stop_on_generalize=false: every seed trains all {epochs} epochs even after "
                "reaching 100% test accuracy",
                "deliberate for a trajectory study; set optim.stop_on_generalize=true to stop early",
            )
        )
    return checks


def check_sage(
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> Check:
    """Sage/GAP, needed only to regenerate group artifacts."""
    try:
        result = run(["sage", "-c", "print('OK')"], capture_output=True, text=True, timeout=120)
    except FileNotFoundError:
        return Check(
            "sage",
            "FAIL",
            "no `sage` on PATH",
            "uv sync --extra sage (needs Python >= 3.11), or install upstream SageMath",
        )
    except subprocess.TimeoutExpired:
        return Check("sage", "FAIL", "`sage -c` timed out after 120s", "check the Sage install")
    if "OK" not in result.stdout:
        return Check(
            "sage",
            "FAIL",
            f"`sage -c` did not run: {result.stderr[:200]}",
            "uv sync --extra sage",
        )
    return Check("sage", "OK", "sage runs")


def run_checks(mode: Mode, overrides: Sequence[str], seeds: Sequence[int] | None) -> list[Check]:
    checks = [check_python()]
    if checks[0].status == "FAIL":
        return checks
    if mode == "export":
        return [*checks, check_sage()]
    config_check, config = check_config(overrides)
    checks.append(config_check)
    if config is None:
        return checks
    checks.extend(check_plan(config, seeds))
    checks.extend(check_cuda(config))
    checks.append(check_artifacts(config))
    checks.append(check_disk(seeds))
    checks.append(check_wandb(config))
    return checks


def report(mode: Mode, checks: Sequence[Check]) -> int:
    """Print the checks and return the exit code."""
    print(f"=== pre-flight: {mode} ===\n")
    width = max(len(check.name) for check in checks)
    for check in checks:
        print(f"{check.status:<4}  {check.name:<{width}}  {check.detail}")
    failures = [check for check in checks if check.status == "FAIL"]
    warnings = [check for check in checks if check.status == "WARN"]
    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED — this machine is not ready to {mode}:")
        for check in failures:
            print(f"  {check.name}: {check.detail}")
            if check.fix:
                print(f"    -> {check.fix}")
        return 1
    print(f"All checks passed ({len(warnings)} warning(s)) — ready to {mode}.")
    for check in warnings:
        print(f"  warning: {check.name}: {check.detail}")
        if check.fix:
            print(f"    -> {check.fix}")
    return 0


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--mode",
        choices=["train", "export"],
        default="train",
        help="train: what a GPU host needs. export: what regenerating artifacts with Sage needs.",
    )
    parser.add_argument(
        "--seeds",
        help="Seeds the campaign will run, e.g. 0-19. Sizes the disk check.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        dest="override",
        help="A Hydra override, repeatable. Spelled exactly as the campaign's own "
        "--override (scripts/run_campaign.py), so the same flags can be copied "
        "between the two commands and preflight checks the config the campaign runs.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra overrides, e.g. data.group.order=32 logging.mode=online",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_arguments(argv)
    overrides = [*args.override, *args.overrides]
    seeds = parse_seeds(args.seeds) if args.seeds else None
    return report(args.mode, run_checks(args.mode, overrides, seeds))


if __name__ == "__main__":
    sys.exit(main())
