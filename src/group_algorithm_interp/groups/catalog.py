"""Canonical SmallGroups artifact lookup."""

from __future__ import annotations

from group_algorithm_interp.groups.data import GroupData, load_group

# Convenience name → SmallGroup(order, index) mapping for common groups.
# The authoritative identity is always the SmallGroup ID; these names are
# just testing/config shorthands.  If you add a name, export its artifact first:
#   python scripts/export_group.py --order <o> --index <i>
_NAMED: dict[str, tuple[int, int]] = {
    "C2": (2, 1),
    "C3": (3, 1),
    "C4": (4, 1),
    "C5": (5, 1),
    "C6": (6, 2),
    "C7": (7, 1),
    "C8": (8, 1),
    "S3": (6, 1),
    "D8": (8, 3),
    "Q8": (8, 4),
}


def build_from_id(gap_id: tuple[int, int] | str) -> GroupData:
    if isinstance(gap_id, str):
        if gap_id in _NAMED:
            return load_group(*_NAMED[gap_id])
        try:
            order_text, index_text = gap_id.split(",")
            gap_id = (int(order_text), int(index_text))
        except ValueError as error:
            raise ValueError(
                f"Unrecognised group identifier {gap_id!r}. "
                "Use a named shorthand (e.g. 'S3', 'Q8') or a canonical "
                "'order,index' SmallGroup ID."
            ) from error
    return load_group(*gap_id)


def resolve_group(spec: object) -> GroupData:
    if isinstance(spec, str) and spec in _NAMED:
        return load_group(*_NAMED[spec])
    if isinstance(spec, str) and "," in spec:
        return build_from_id(spec)
    if hasattr(spec, "order") and hasattr(spec, "index"):
        return load_group(int(getattr(spec, "order")), int(getattr(spec, "index")))
    return build_from_id(spec)  # type: ignore[arg-type]


__all__ = ["build_from_id", "resolve_group"]
