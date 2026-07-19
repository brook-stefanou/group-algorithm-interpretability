"""Guard against config fields that silently do nothing.

Every field on ``ProjectConfig`` (and every nested sub-config it composes) must
be *read* somewhere in the runtime source -- ``src/``, ``scripts/``, or
``config.py`` itself (e.g. the ``model_validator`` that derives
``optim.lr_effective`` from ``optim.scale_lr_with_width``). A field only ever
declared and defaulted, never consulted by any runtime code including its own
validators, means setting it has no effect: the silent no-op this project's
config validation exists to prevent (see the module docstring in
``config.py``).

What counts as a read
---------------------
The corpus is parsed with ``ast``, not scanned as text, and evidence for a field
is one of:

* an attribute load -- ``cfg.epochs``, ``self.config.optim.epochs``;
* a string-subscript load -- ``prediction["metric"]`` (how ``Prediction``'s
  fields are read after ``model_dump()``);
* a constant string handed to ``getattr``/``hasattr`` -- ``getattr(spec, "order")``
  (how ``catalog.resolve_group`` reads a ``GroupSpec``).

Comments and string literals (including docstrings) are *not* part of the
corpus, and neither is a bare mention of the name. A prior version of this
guard asked ``field_name in <raw file text>``, and that substring test passed
on two genuinely dead fields:

* ``experiment.steps`` -- kept "alive" only by a comment in ``config.py`` that
  was itself wrong. Nothing read it, so ``experiment=debug`` and
  ``experiment=smoke`` did not shorten a run: all three presets trained for
  ``optim.epochs`` = 10,000.
* ``eval.metric`` -- whose *default value* was the literal string ``"metric"``,
  so the field name appeared in its own declaration's default.

An assignment target (``self.optim.lr_effective = ...``) is a store, not a load,
and does not on its own count: a value written and never read is dead in exactly
the way this guard exists to catch. ``lr_effective`` passes because
``experiment.py`` loads it to build the optimiser.

Known limitation: evidence is collected by field *name*, so a field whose name
is also declared on another config model is masked by that other model's reads.
``test_field_name_shadowing_is_documented`` pins the set of shadowed names, so a
newly introduced collision (the condition that hid ``eval.metric`` behind
``Prediction.metric``) fails and has to be looked at rather than being absorbed.
"""

from __future__ import annotations

import ast
import typing
from collections import defaultdict
from pathlib import Path

import pytest
from pydantic import BaseModel

from group_algorithm_interp import config as config_module
from group_algorithm_interp.config import ProjectConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = REPO_ROOT / "scripts"
CONFIG_PY = Path(config_module.__file__).resolve()

# Fields that are genuinely unread anywhere in the runtime source (src/,
# scripts/, and config.py's own validators/methods) but cannot be removed
# right now. Keep this empty -- every entry is a debt, not a convenience, and
# must name the concrete blocker.
ALLOWLIST: dict[str, str] = {}

# Field names declared on more than one config model. Evidence is name-scoped,
# so for these the guard cannot tell which model's field a read belongs to: each
# one has to be justified by hand here. A collision that is NOT in this table is
# a test failure -- that is how a dead field last hid in plain sight.
SHADOWED_FIELD_NAMES: dict[str, str] = {
    "group": (
        "DataConfig.group (the finite group; read in experiment.py and manifest.py) "
        "and LoggingConfig.group (the W&B group; read in wandb_utils.py). Both are "
        "independently, genuinely read -- the collision is only in the name."
    ),
    "seed": (
        "ProjectConfig.seed (the run's seed; read in experiment.py and config.py) "
        "and ExperimentConfig.seed (the constructor-only alias that overwrites it; "
        "read in ProjectConfig._reconcile_and_guard). Both are genuinely read."
    ),
}


def _unwrap_annotation(annotation: object) -> list[object]:
    """Flatten Optional[...]/Union[...]/list[...]/tuple[...] wrappers down to
    their leaf type arguments (plus the annotation itself when it has none)."""
    origin = typing.get_origin(annotation)
    if origin is None:
        return [annotation]
    leaves: list[object] = []
    for arg in typing.get_args(annotation):
        leaves.extend(_unwrap_annotation(arg))
    return leaves


def _iter_model_classes(
    model: type[BaseModel], seen: set[type[BaseModel]] | None = None
) -> typing.Iterator[type[BaseModel]]:
    if seen is None:
        seen = set()
    if model in seen:
        return
    seen.add(model)
    yield model
    for field in model.model_fields.values():
        for candidate in _unwrap_annotation(field.annotation):
            if isinstance(candidate, type) and issubclass(candidate, BaseModel):
                yield from _iter_model_classes(candidate, seen)


def _fields_by_name() -> dict[str, list[str]]:
    """field name -> the config models that declare it."""
    owners: dict[str, list[str]] = defaultdict(list)
    for model_cls in _iter_model_classes(ProjectConfig):
        for field_name in model_cls.model_fields:
            owners[field_name].append(model_cls.__name__)
    return dict(owners)


class _ReadCollector(ast.NodeVisitor):
    """Names a module genuinely *reads* as attributes / string keys."""

    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # Loads only: `x.field` reads it, `x.field = ...` (Store) does not.
        if isinstance(node.ctx, ast.Load):
            self.names.add(node.attr)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.ctx, ast.Load) and isinstance(node.slice, ast.Constant):
            if isinstance(node.slice.value, str):
                self.names.add(node.slice.value)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # getattr(spec, "order") / hasattr(spec, "order") are real reads too.
        if (
            isinstance(node.func, ast.Name)
            and node.func.id in {"getattr", "hasattr"}
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            self.names.add(node.args[1].value)
        self.generic_visit(node)


def _read_names() -> set[str]:
    """Every name read by runtime code under src/ and scripts/ (config.py
    included: its validators and properties are runtime code too)."""
    collector = _ReadCollector()
    for base in (SRC_DIR, SCRIPTS_DIR):
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            collector.visit(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
    return collector.names


_READS = _read_names()
_FIELD_OWNERS = _fields_by_name()


@pytest.mark.parametrize("field_name", sorted(_FIELD_OWNERS))
def test_config_field_is_read_somewhere(field_name: str) -> None:
    if field_name in ALLOWLIST:
        pytest.skip(f"allowlisted: {ALLOWLIST[field_name]}")
    owners = ", ".join(_FIELD_OWNERS[field_name])
    assert field_name in _READS, (
        f"config field {field_name!r} (declared on {owners}) is never read in "
        "src/ or scripts/ -- no attribute access, no string-key lookup, no "
        "getattr. A mention in a comment, a docstring, or its own default value "
        "does not count. Setting it currently does nothing: either wire it up, "
        "remove it from the schema, or add a justified ALLOWLIST entry here."
    )


def test_allowlist_entries_still_exist_as_fields() -> None:
    """Catch a stale allowlist entry for a field that was since removed."""
    stale = set(ALLOWLIST) - set(_FIELD_OWNERS)
    assert not stale, f"allowlist references fields no longer in the schema: {stale}"


def test_field_name_shadowing_is_documented() -> None:
    """A field name declared on two config models makes the name-scoped guard
    blind: one model's reads vouch for the other's field. Every such collision
    must be justified in SHADOWED_FIELD_NAMES, so a new one cannot quietly
    resurrect the eval.metric failure mode."""
    shadowed = {name for name, owners in _FIELD_OWNERS.items() if len(owners) > 1}
    undocumented = {name: _FIELD_OWNERS[name] for name in shadowed - set(SHADOWED_FIELD_NAMES)}
    assert not undocumented, (
        "config field names declared on more than one model, with no entry in "
        f"SHADOWED_FIELD_NAMES: {undocumented}. The dead-field guard is "
        "name-scoped, so these fields vouch for each other: verify that EACH is "
        "really read, then document the collision (or rename a field)."
    )
    stale = set(SHADOWED_FIELD_NAMES) - shadowed
    assert not stale, (
        f"SHADOWED_FIELD_NAMES documents names that no longer collide: {stale}. "
        "Remove the entry -- the guard can see these fields on its own now."
    )
