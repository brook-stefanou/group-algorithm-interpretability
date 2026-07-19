"""Tests for training/trainer.py: snapshot policy, model construction, and helpers."""

from __future__ import annotations

import pytest
import torch

from group_algorithm_interp.config import ProjectConfig, SnapshotConfig
from group_algorithm_interp.groups.data import load_group
from group_algorithm_interp.model import FCModel, OneLayerTransformer
from group_algorithm_interp.training.trainer import _is_power_of_two, build_model, should_snapshot

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def group_c8():
    """SmallGroup(8, 1) -- C8, a well-known artifact that always exists."""
    return load_group(8, 1)


# ---------------------------------------------------------------------------
# _is_power_of_two
# ---------------------------------------------------------------------------


class TestIsPowerOfTwo:
    @pytest.mark.parametrize(
        "value",
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
    )
    def test_powers_of_two(self, value: int) -> None:
        assert _is_power_of_two(value) is True

    @pytest.mark.parametrize(
        "value",
        [0, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18, 20, 21],
    )
    def test_non_powers_of_two(self, value: int) -> None:
        assert _is_power_of_two(value) is False

    @pytest.mark.parametrize(
        "value",
        [-1, -2, -4, -8, -16, -100],
    )
    def test_negative_values(self, value: int) -> None:
        assert _is_power_of_two(value) is False

    @pytest.mark.parametrize(
        "value",
        [2**20, 2**30, 2**40],
    )
    def test_large_powers_of_two(self, value: int) -> None:
        assert _is_power_of_two(value) is True

    @pytest.mark.parametrize(
        "value",
        [1023, 1025, 2047, 2049, 4095, 4097],
    )
    def test_near_powers_of_two(self, value: int) -> None:
        assert _is_power_of_two(value) is False


# ---------------------------------------------------------------------------
# should_snapshot
# ---------------------------------------------------------------------------
#
# One parametrized sweep over the decision's actual branches: disabled, step
# 0, negative step, dense power-of-two window, the log_dense_until boundary,
# and the post-dense interval check.


class TestShouldSnapshot:
    @staticmethod
    def cfg(
        enabled: bool = True, log_dense_until: int = 1024, interval: int = 1000
    ) -> SnapshotConfig:
        return SnapshotConfig(enabled=enabled, log_dense_until=log_dense_until, interval=interval)

    @pytest.mark.parametrize(
        "step,enabled,log_dense_until,interval,expected",
        [
            (0, True, 1024, 1000, True),  # step 0 always fires when enabled
            (0, True, 0, 7, True),  # step 0 overrides an empty dense window
            (0, False, 1024, 1000, False),  # disabled: even step 0 does not fire
            (1000, False, 1024, 1000, False),  # disabled: nothing fires
            (-1, True, 1024, 1000, False),  # negative step never fires
            (-100, False, 1024, 1000, False),  # negative + disabled
            (16, True, 1024, 1000, True),  # power of two within the dense window
            (17, True, 1024, 1000, False),  # non-power-of-two within the dense window
            (1024, True, 1024, 1000, True),  # exactly at log_dense_until, a power of two
            (1023, True, 1023, 1000, False),  # exactly at log_dense_until, not a power of two
            (2000, True, 1024, 1000, True),  # beyond dense, an exact interval multiple
            (
                2048,
                True,
                1024,
                1000,
                False,
            ),  # beyond dense, power of two but not an interval multiple
            (1025, True, 1024, 1000, False),  # beyond dense, neither
            (1, True, 0, 1000, False),  # log_dense_until=0: no dense window at all
            (1000, True, 0, 1000, True),  # ...falls through to the interval check
            (1026, True, 1024, 1, True),  # interval=1: every step beyond dense fires
        ],
    )
    def test_should_snapshot(
        self, step: int, enabled: bool, log_dense_until: int, interval: int, expected: bool
    ) -> None:
        c = self.cfg(enabled=enabled, log_dense_until=log_dense_until, interval=interval)
        assert should_snapshot(step, c) is expected


def test_project_config_default_snapshot_values() -> None:
    """Pins ProjectConfig's default SnapshotConfig field values -- the
    defaults an operator gets from an unconfigured run, not a re-test of
    should_snapshot's own logic (covered above)."""
    snap = ProjectConfig(seed=1).snapshot
    assert (snap.enabled, snap.log_dense_until, snap.interval) == (True, 1024, 1000)


# ---------------------------------------------------------------------------
# build_model
# ---------------------------------------------------------------------------
#
# These test build_model's own contract -- vocab sizing, n_ctx, and config ->
# architecture/field propagation -- not the model's forward behaviour, which
# is test_model.py's and test_fc_model.py's job.


class TestBuildModel:
    @staticmethod
    def _default_config(**kwargs: object) -> ProjectConfig:
        return ProjectConfig(seed=1, **kwargs)  # type: ignore[arg-type]

    # -- architecture dispatch --------------------------------------------------

    def test_default_arch_is_transformer(self, group_c8) -> None:
        cfg = self._default_config()
        model = build_model(cfg, group_c8)
        assert isinstance(model, OneLayerTransformer)

    def test_fc_arch(self, group_c8) -> None:
        cfg = self._default_config(model={"arch": "fc"})
        model = build_model(cfg, group_c8)
        assert isinstance(model, FCModel)

    def test_transformer_arch_explicit(self, group_c8) -> None:
        cfg = self._default_config(model={"arch": "transformer"})
        model = build_model(cfg, group_c8)
        assert isinstance(model, OneLayerTransformer)

    # -- vocabulary sizes -------------------------------------------------------

    def test_transformer_d_vocab(self, group_c8) -> None:
        cfg = self._default_config()
        model = build_model(cfg, group_c8)
        assert model.d_vocab_in == group_c8.order + 1  # extra token for '='
        assert model.d_vocab_out == group_c8.order

    def test_fc_d_vocab(self, group_c8) -> None:
        cfg = self._default_config(model={"arch": "fc"})
        model = build_model(cfg, group_c8)
        assert model.d_vocab_in == group_c8.order + 1
        assert model.d_vocab_out == group_c8.order

    # -- config values propagated -----------------------------------------------

    def test_transformer_d_model(self, group_c8) -> None:
        cfg = self._default_config(model={"d_model": 128})
        model = build_model(cfg, group_c8)
        assert model.d_model == 128

    def test_transformer_n_heads(self, group_c8) -> None:
        cfg = self._default_config(model={"n_heads": 8, "d_model": 64})
        model = build_model(cfg, group_c8)
        assert model.n_heads == 8

    def test_transformer_d_mlp(self, group_c8) -> None:
        cfg = self._default_config(model={"d_mlp": 512})
        model = build_model(cfg, group_c8)
        assert model.d_mlp == 512

    def test_transformer_activation(self, group_c8) -> None:
        cfg = self._default_config(model={"activation": "gelu"})
        model = build_model(cfg, group_c8)
        assert isinstance(model.activation, torch.nn.GELU)

    def test_fc_d_model(self, group_c8) -> None:
        cfg = self._default_config(model={"arch": "fc", "d_model": 128})
        model = build_model(cfg, group_c8)
        assert model.d_model == 128

    def test_fc_d_mlp(self, group_c8) -> None:
        cfg = self._default_config(model={"arch": "fc", "d_mlp": 512})
        model = build_model(cfg, group_c8)
        assert model.d_mlp == 512

    def test_fc_activation(self, group_c8) -> None:
        cfg = self._default_config(model={"arch": "fc", "activation": "silu"})
        model = build_model(cfg, group_c8)
        assert isinstance(model.activation, torch.nn.SiLU)

    # -- use_mlp off ------------------------------------------------------------

    def test_transformer_no_mlp(self, group_c8) -> None:
        cfg = self._default_config(model={"use_mlp": False, "n_heads": 4, "d_model": 64})
        model = build_model(cfg, group_c8)
        assert model.use_mlp is False
        assert not hasattr(model, "W_in")
        assert not hasattr(model, "W_out")

    def test_transformer_with_mlp(self, group_c8) -> None:
        cfg = self._default_config(model={"use_mlp": True, "n_heads": 4, "d_model": 64})
        model = build_model(cfg, group_c8)
        assert model.use_mlp is True
        assert hasattr(model, "W_in")
        assert hasattr(model, "W_out")

    # -- n_ctx is always 3 ------------------------------------------------------

    def test_transformer_n_ctx(self, group_c8) -> None:
        cfg = self._default_config()
        model = build_model(cfg, group_c8)
        assert model.n_ctx == 3

    def test_fc_n_ctx(self, group_c8) -> None:
        cfg = self._default_config(model={"arch": "fc"})
        model = build_model(cfg, group_c8)
        assert model.n_ctx == 3


# ---------------------------------------------------------------------------
# build_model with different groups
# ---------------------------------------------------------------------------


class TestBuildModelWithGroups:
    def test_different_group_order(self) -> None:
        g21 = load_group(21, 2)
        cfg = ProjectConfig(seed=1)
        model = build_model(cfg, g21)
        assert model.d_vocab_in == 22
        assert model.d_vocab_out == 21

    def test_transformer_no_mlp_different_group(self) -> None:
        g21 = load_group(21, 2)
        cfg = ProjectConfig(seed=1, model={"use_mlp": False, "n_heads": 4, "d_model": 64})
        model = build_model(cfg, g21)
        assert model.use_mlp is False
        assert not hasattr(model, "W_in")
        assert not hasattr(model, "W_out")
