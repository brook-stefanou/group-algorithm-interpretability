"""Compatibility names for the Sage/GAP model adapter (no algebra implementation)."""

from group_algorithm_interp.groups.data import GroupData

Element = int
FiniteGroup = GroupData

__all__ = ["Element", "FiniteGroup"]
