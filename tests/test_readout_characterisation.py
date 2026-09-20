"""The positive readout characterisation (instruments/readout_characterisation.py):
character-only vs full-matrix-entry, held out.

Two synthetic-control tests carry the module's whole claim:

* Logits that ARE exactly the character readout (``Re tr(rho(a b)
  rho(c^-1))``) over one irrep, plus noise, must show the full-matrix-entry
  design (a strict, genuinely nested superset) buying essentially nothing held
  out -- ``character_only_sufficient`` True, the positive result.
* Logits that need a specific *off-diagonal* entry of the un-contracted matrix
  product ``M(a, b, c) = rho(a b) @ rho(c^-1)`` -- not expressible as any
  multiple of its trace -- must show the full-matrix-entry design winning held
  out -- ``character_only_sufficient`` False.

Plus a direct correctness check of the nesting property itself
(``Re tr(M) == sum of the (Re, i == j) columns``, the algebraic fact the whole
instrument leans on) and the vacuous-group guard.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from group_algorithm_interp.groups.catalog import resolve_group
from group_algorithm_interp.instruments import probes as P
from group_algorithm_interp.instruments import readout_characterisation as R


class _FakeModel:
    """A model stub whose read-position logits are a fixed ``[order, order, C]``
    grid -- lets the I-20 harness be tested on logits of a known functional
    form without training a real model. Matches ``tests/test_probes.py``'s
    fixture of the same name."""

    def __init__(self, logits_grid: np.ndarray):
        self._grid = torch.tensor(logits_grid, dtype=torch.float64)
        self.W_E = torch.zeros(logits_grid.shape[0] + 1, 2)

    def eval(self) -> None:  # noqa: D401 - matches nn.Module.eval()
        return None

    def __call__(self, tokens: torch.Tensor) -> torch.Tensor:
        order = self._grid.shape[0]
        flat = self._grid.reshape(order * order, -1)
        return flat.unsqueeze(1).expand(-1, 3, -1)


def _std_irrep_index(group) -> int:
    candidates = P.gcr_candidate_irreps(group)
    return next(i for i in candidates if group.irreps[i].dimension == 2)


# ---------------------------------------------------------------------------
# Nesting correctness: the algebraic fact the whole comparison leans on.
# ---------------------------------------------------------------------------


def test_character_design_equals_sum_of_diagonal_real_columns_of_matrix_entry_design():
    group = resolve_group("S3")
    std_idx = _std_irrep_index(group)
    character = P.gcr_character_design(group, [std_idx])[..., 0]
    matrix_entries = R.gcr_matrix_entry_design(group, [std_idx])
    d = group.irreps[std_idx].dimension
    # Real block is the first d*d columns, row-major (i, j); the diagonal
    # entries are indices i * d + i.
    diagonal_real = sum(matrix_entries[..., i * d + i] for i in range(d))
    assert np.allclose(character, diagonal_real)


def test_matrix_entry_design_matches_direct_computation():
    """The design column at a fixed (a, b, c, i, j) is exactly
    ``[rho(a b) rho(c^-1)]_{ij}``, computed directly -- not merely
    self-consistent with the vectorised implementation."""
    group = resolve_group("S3")
    std_idx = _std_irrep_index(group)
    design = R.gcr_matrix_entry_design(group, [std_idx])
    matrices = group.irreps[std_idx].matrices
    table = group.cayley_table
    inv = P.inverses(table, P.identity_index(table))
    d = matrices.shape[1]
    for a, b, c in [(2, 4, 5), (0, 1, 2), (5, 5, 0)]:
        ab = int(table[a, b])
        expected = matrices[ab] @ matrices[int(inv[c])]
        for i in range(d):
            for j in range(d):
                assert design[a, b, c, i * d + j] == pytest.approx(expected[i, j].real)
                assert design[a, b, c, d * d + i * d + j] == pytest.approx(expected[i, j].imag)


# ---------------------------------------------------------------------------
# Vacuous-group guard
# ---------------------------------------------------------------------------


def test_undefined_when_group_has_no_nontrivial_irrep(monkeypatch):
    group = resolve_group("S3")
    monkeypatch.setattr(R, "gcr_candidate_irreps", lambda g: [])
    model = _FakeModel(np.zeros((group.order, group.order, group.order)))
    record = R.readout_characterisation_instrument(model, group, seed=0)
    assert record["status"] == R.UNDEFINED


# ---------------------------------------------------------------------------
# Synthetic control 1: logits ARE the character -- full-matrix-entry ties.
# ---------------------------------------------------------------------------


def test_character_only_ties_full_matrix_entry_when_logits_are_the_character():
    group = resolve_group("S3")
    std_idx = _std_irrep_index(group)
    character = P.gcr_character_design(group, [std_idx])[..., 0]
    rng = np.random.default_rng(0)
    logits = character + rng.normal(scale=0.01, size=character.shape)
    model = _FakeModel(logits)

    record = R.readout_characterisation_instrument(model, group, seed=0)

    assert record["status"] == "measured"
    assert record["character_only_held_out_fve"] > 0.9
    # The full-matrix-entry design is a strict superset, so it cannot fit
    # worse in sample, but on held-out cells its extra (genuinely
    # unnecessary) columns buy nothing meaningful over the character alone.
    assert record["held_out_fve_gain_full_minus_character"] <= record["tie_tol"]
    assert record["character_only_sufficient"] is True
    assert record["minimal_irrep_set"]["selected_irrep_indices"] == [std_idx]


# ---------------------------------------------------------------------------
# Synthetic control 2: logits need an off-diagonal matrix entry -- full wins.
# ---------------------------------------------------------------------------


def test_full_matrix_entry_wins_when_logits_need_an_off_diagonal_entry():
    group = resolve_group("S3")
    std_idx = _std_irrep_index(group)
    matrix_entries = R.gcr_matrix_entry_design(group, [std_idx])
    d = group.irreps[std_idx].dimension
    # Column index 1 is Re(M_{0,1}) (row-major (i, j) = (0, 1)): a genuine
    # off-diagonal entry, not expressible as any multiple of the trace
    # Re(M_00) + Re(M_11).
    off_diagonal = matrix_entries[..., 1]
    assert d == 2  # column-index arithmetic above assumes the 2-D standard irrep
    rng = np.random.default_rng(1)
    logits = off_diagonal + rng.normal(scale=0.01, size=off_diagonal.shape)
    model = _FakeModel(logits)

    record = R.readout_characterisation_instrument(model, group, seed=0)

    assert record["status"] == "measured"
    assert record["full_matrix_entry_held_out_fve"] > 0.9
    assert record["character_only_held_out_fve"] < 0.5
    assert record["held_out_fve_gain_full_minus_character"] > record["tie_tol"]
    assert record["character_only_sufficient"] is False


# ---------------------------------------------------------------------------
# Null-model convention (I-20's rule-1 regression, reused unchanged).
# ---------------------------------------------------------------------------


def test_null_model_scores_reported_when_supplied():
    group = resolve_group("S3")
    std_idx = _std_irrep_index(group)
    character = P.gcr_character_design(group, [std_idx])[..., 0]
    rng = np.random.default_rng(0)
    logits = character + rng.normal(scale=0.01, size=character.shape)
    model = _FakeModel(logits)
    null_logits = rng.normal(scale=1.0, size=character.shape)
    null_model = _FakeModel(null_logits)

    record = R.readout_characterisation_instrument(model, group, null_model=null_model, seed=0)

    assert "null" in record
    assert record["null"]["character_only_null_held_out_fve"] < 0.3
    assert record["null"]["full_matrix_entry_null_held_out_fve"] < 0.3
