"""Accessors for the representation arrays Sage/GAP exports as ground truth."""

from group_algorithm_interp.groups.data import GroupData


def compute_character_table(group: GroupData):
    return group.character_table, [list(cls) for cls in group.conjugacy_classes]


def frobenius_schur_indicators(group: GroupData):
    return group.frobenius_schur


def extract_irreps(group: GroupData):
    return list(group.irreps)


def isotypic_projectors(group: GroupData):
    return [block.projector for block in group.isotypic_blocks]


def real_isotypic_blocks(group: GroupData):
    return list(group.isotypic_blocks)


__all__ = [
    "compute_character_table",
    "extract_irreps",
    "frobenius_schur_indicators",
    "isotypic_projectors",
    "real_isotypic_blocks",
]
