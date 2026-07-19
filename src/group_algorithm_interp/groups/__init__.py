"""Sage/GAP-backed, serialisable finite-group adapters."""

from .data import GroupData, IrrepData, IsotypicBlock, load_group
from .tensor_rank import (
    group_algebra_tensor_rank_bounds,
    irrep_rank_lower_bound,
    irrep_rank_upper_bound,
)

__all__ = [
    "GroupData",
    "IrrepData",
    "IsotypicBlock",
    "load_group",
    "group_algebra_tensor_rank_bounds",
    "irrep_rank_lower_bound",
    "irrep_rank_upper_bound",
]
