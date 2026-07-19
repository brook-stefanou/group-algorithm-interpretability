"""Project-specific model construction and snapshot policy.

Run lifecycle responsibilities deliberately remain in ``BaseExperiment``.
"""

from __future__ import annotations

from ..config import ProjectConfig, SnapshotConfig
from ..groups.group import FiniteGroup
from ..model import FCModel, GroupModel, OneLayerTransformer
from ..seed import set_seed


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def should_snapshot(step: int, config: SnapshotConfig) -> bool:
    if not config.enabled or step < 0:
        return False
    if step == 0:
        return True
    if step <= config.log_dense_until and _is_power_of_two(step):
        return True
    return step % config.interval == 0


def build_model(config: ProjectConfig, group: FiniteGroup) -> GroupModel:
    if config.model.arch == "fc":
        return FCModel(
            d_vocab_in=group.order + 1,
            d_vocab_out=group.order,
            n_ctx=3,
            d_model=config.model.d_model,
            d_mlp=config.model.d_mlp,
            activation=config.model.activation,
        )
    return OneLayerTransformer(
        d_vocab_in=group.order + 1,
        d_vocab_out=group.order,
        n_ctx=3,
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        use_mlp=config.model.use_mlp,
        d_mlp=config.model.d_mlp,
        activation=config.model.activation,
    )


__all__ = ["build_model", "set_seed", "should_snapshot"]
