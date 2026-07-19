"""Export one canonical GAP SmallGroup as a PyTorch-facing Sage artifact.

Usage::

    sage -python scripts/export_group.py --order 8 --index 3   # upstream SageMath
    python scripts/export_group.py --order 8 --index 3          # passagemath via pip

The ordinary Python package never imports Sage or GAP.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _require_sage() -> tuple[Any, Any, Any, str]:
    """Return ``(CC, ComplexField, libgap, version_string)``.

    Tries the monolithic ``sage.all`` namespace first (upstream SageMath and
    ``passagemath-standard``), then falls back to the *modular* passagemath
    namespaces.  The fallback is what makes the minimal extra viable: modular
    passagemath deliberately does not ship ``sage.all``, so importing it is the
    only way to run under ``passagemath-gap`` + ``passagemath-modules`` alone
    (51 distributions) rather than pulling in all of ``passagemath-standard``
    (172).  ``libgap`` lives in the -gap distribution and the MPFR complex
    field (``CC``/``ComplexField``) in -modules; neither implies the other, so
    both are imported explicitly and a missing one fails loudly here rather
    than midway through an export.
    """
    # --- primary path: sage.all (upstream SageMath and passagemath-standard) ---
    try:
        import sage.version
        from sage.all import CC, ComplexField, libgap

        return CC, ComplexField, libgap, sage.version.version
    except ImportError:
        pass

    # --- fallback: modular passagemath (no sage.all by design) ---
    try:
        import sage.version
        from sage.all__sagemath_gap import libgap
        from sage.all__sagemath_modules import CC, ComplexField

        return CC, ComplexField, libgap, sage.version.version
    except ImportError:
        pass

    raise SystemExit(
        "SageMath is required. Install it via one of:\n"
        "  uv sync --extra sage                   (pip-installable, recommended)\n"
        "  sage -python scripts/export_group.py …  (upstream SageMath)\n"
        "See https://doc.sagemath.org/html/en/installation/index.html"
    )


def _as_complex(value: Any, cc: Any) -> complex:
    number = cc(value)
    return complex(float(number.real()), float(number.imag()))


def _matrix_values(matrix: Any, cc: Any) -> list[list[complex]]:
    rows = list(matrix)
    return [[_as_complex(value, cc) for value in list(row)] for row in rows]


def _subgroup_section(
    libgap: Any,
    group: Any,
    token_of: dict[str, int],
    *,
    include_subgroups: bool,
) -> tuple[int, list[int], dict[str, Any]]:
    """Build the optional subgroup/left-coset artifact section.

    Gated OFF by default.  Elementary-abelian 2-groups have Gaussian-binomial
    subgroup counts: ``C2^7`` = ``SmallGroup(128, 2328)`` has 29,212 subgroups
    and 387,987 left cosets, and emitting one ``.npz`` member per subgroup and
    per coset would produce a single archive holding ~417,000 zip members (and
    spend ~16 s enumerating the lattice).  Nothing under ``src/`` consumes coset
    data yet, so the default export omits the whole section and records
    ``subgroups: 0`` / ``coset_counts: []`` in the metadata.  Pass
    ``include_subgroups=True`` only for the specific groups whose ``Ind_H^G 1``
    coset templates an instrument actually needs.

    When ``include_subgroups`` is ``False`` no GAP enumeration is performed, so
    ``libgap`` and ``group`` are never dereferenced on that path (the flag
    behaviour is therefore testable without Sage installed).
    """
    import numpy as np

    if not include_subgroups:
        return 0, [], {}

    subgroup_tokens = []
    cosets = []
    for subgroup in libgap.AllSubgroups(group):
        subgroup_elements = list(libgap.Elements(subgroup))
        subgroup_tokens.append(
            np.asarray(sorted(token_of[str(x)] for x in subgroup_elements), dtype=np.int64)
        )
        cosets.append(
            [
                np.asarray(sorted(token_of[str(x)] for x in libgap.AsList(coset)), dtype=np.int64)
                for coset in libgap.LeftCosets(group, subgroup)
            ]
        )
    payload: dict[str, Any] = {}
    for i, subgroup in enumerate(subgroup_tokens):
        payload[f"subgroup_{i}"] = subgroup
        for j, coset in enumerate(cosets[i]):
            payload[f"coset_{i}_{j}"] = coset
    return len(subgroup_tokens), [len(items) for items in cosets], payload


def export(order: int, index: int, output: Path, *, include_subgroups: bool = False) -> Path:
    import numpy as np

    CC, _complex_field, libgap, sage_version = _require_sage()
    group = libgap.SmallGroup(order, index)
    elements = list(libgap.Elements(group))
    labels = [str(element) for element in elements]
    if len(set(labels)) != len(labels):
        raise RuntimeError(
            "GAP element string labels are not unique; cannot create stable token mapping"
        )
    token_of = {label: token for token, label in enumerate(labels)}
    n = len(elements)
    table = np.empty((n, n), dtype=np.int64)
    for i, left in enumerate(elements):
        for j, right in enumerate(elements):
            table[i, j] = token_of[str(left * right)]

    classes = [list(libgap.AsList(cls)) for cls in libgap.ConjugacyClasses(group)]
    class_tokens = [np.array([token_of[str(x)] for x in cls], dtype=np.int64) for cls in classes]
    class_of = np.empty(n, dtype=np.int64)
    for class_index, cls in enumerate(class_tokens):
        class_of[cls] = class_index

    characters = list(libgap.Irr(group))
    # GAP's character order is authoritative. It is retained verbatim so rows,
    # projectors, FS indicators, and concrete representations stay aligned.
    char_rows = []
    for character in characters:
        values = list(libgap.ValuesOfClassFunction(character))
        char_rows.append([_as_complex(value, CC) for value in values])
    char_table = np.asarray(char_rows, dtype=np.complex128)
    # Frobenius-Schur indicators: FS(χ) = (1/|G|) Σ_{g∈G} χ(g²)
    # Computed from the character table so we don't depend on GAP's ctbllib Indicators.
    square_tokens = np.array([table[g, g] for g in range(n)], dtype=np.int64)
    fs_list = []
    for row in range(len(characters)):
        chi_g2 = char_table[row, class_of[square_tokens]]
        fs_val = np.sum(chi_g2).real / n
        fs_list.append(int(round(fs_val)))
    fs = np.asarray(fs_list, dtype=np.int64)

    try:
        reps = list(libgap.IrreducibleRepresentations(group))
    except Exception as error:
        raise RuntimeError(
            "GAP could not provide concrete irreducible representations for "
            f"SmallGroup({order},{index}); no incomplete artifact was written."
        ) from error
    if len(reps) != len(characters):
        raise RuntimeError("GAP representation and character counts disagree")
    irrep_stacks: list[np.ndarray | None] = [None] * len(characters)
    # GAP may return Irr(G) and IrreducibleRepresentations(G) in different
    # orders (especially under passagemath).  Match each representation to its
    # character row by comparing element-wise traces.
    for rep_idx, representation in enumerate(reps):
        matrices = []
        for element in elements:
            image = libgap.Image(representation, element)
            matrices.append(np.asarray(_matrix_values(image, CC), dtype=np.complex128))
        stack = np.asarray(matrices, dtype=np.complex128)
        traces = np.trace(stack, axis1=1, axis2=2)
        found = False
        for char_row in range(len(characters)):
            if irrep_stacks[char_row] is not None:
                continue
            expected = char_table[char_row, class_of]
            if np.allclose(traces, expected, atol=1e-10):
                irrep_stacks[char_row] = stack
                found = True
                break
        if not found:
            raise RuntimeError(
                f"GAP representation {rep_idx} could not be matched to any character row."
            )
    if any(m is None for m in irrep_stacks):
        missing = [i for i, m in enumerate(irrep_stacks) if m is None]
        raise RuntimeError(f"Character rows without a matching representation: {missing}")

    # Exact Sage/GAP character data determines these projectors; numerical
    # conversion happens only after the sum.  Complex conjugates are combined
    # into real blocks for the existing tensor-facing probes.
    regular = np.zeros((n, n, n), dtype=np.complex128)
    for g in range(n):
        regular[g, table[g], np.arange(n)] = 1.0
    individual = []
    for row, character in enumerate(char_table):
        degree = int(round(character[0].real))
        individual.append(
            sum(np.conj(character[class_of[g]]) * regular[g] for g in range(n)) * degree / n
        )
    used: set[int] = set()
    block_specs: list[dict[str, Any]] = []
    block_projectors: list[np.ndarray] = []
    for row, character in enumerate(char_table):
        if row in used:
            continue
        partner = next(
            (
                other
                for other in range(row + 1, len(char_table))
                if np.allclose(char_table[other], np.conj(character))
            ),
            None,
        )
        rows = (row,) if partner is None else (row, partner)
        used.update(rows)
        projector = sum((individual[item] for item in rows), np.zeros((n, n), dtype=np.complex128))
        if not np.allclose(projector.imag, 0.0, atol=1e-10):
            raise RuntimeError(f"isotypic block {rows} did not convert to a real projector")
        block_projectors.append(projector.real)
        # Two distinct integers.  ``irrep_degree`` is chi(e); both members of a
        # conjugate pair share it.  ``block_rank`` is the dimension of the
        # isotypic component in the regular representation: an irrep of degree d
        # appears with multiplicity d, hence spans d^2 dimensions, and a merged
        # conjugate pair spans 2*d^2.  It equals trace(P) == rank(P).
        degrees = [int(round(char_table[item, 0].real)) for item in rows]
        irrep_degree = degrees[0]
        if any(degree != irrep_degree for degree in degrees):
            raise RuntimeError(f"isotypic block {rows} merges irreps of different degrees")
        block_rank = sum(degree**2 for degree in degrees)
        trace = float(np.trace(projector.real))
        if abs(trace - block_rank) > 1e-6:
            raise RuntimeError(
                f"isotypic block {rows}: trace(P)={trace:.6f} but sum of d^2 = {block_rank}"
            )
        block_specs.append(
            {
                "irrep_degree": irrep_degree,
                "block_rank": block_rank,
                "irrep_indices": rows,
            }
        )
    total_rank = sum(int(spec["block_rank"]) for spec in block_specs)
    if total_rank != n:
        raise RuntimeError(f"isotypic block ranks sum to {total_rank}, expected |G| = {n}")

    subgroup_count, coset_counts, subgroup_payload = _subgroup_section(
        libgap, group, token_of, include_subgroups=include_subgroups
    )

    payload: dict[str, Any] = {
        "metadata": np.array(
            json.dumps(
                {
                    "format": 2,
                    "small_group": [order, index],
                    "order": n,
                    "description": str(libgap.StructureDescription(group)),
                    "element_labels": labels,
                    "classes": len(class_tokens),
                    "irreps": [
                        {
                            "dimension": int(matrix.shape[1]),  # type: ignore[union-attr]
                            "field": "GAP default field",
                            "basis": "GAP IrreducibleRepresentations basis",
                        }
                        for matrix in irrep_stacks
                    ],
                    "blocks": block_specs,
                    "subgroups": subgroup_count,
                    "coset_counts": coset_counts,
                    "provenance": {
                        "backend": "SageMath/libgap",
                        "sage_version": sage_version,
                        "gap_version": str(libgap.eval("GAPInfo.Version;")),
                        "element_ordering": "GAP Elements(SmallGroup(order,index))",
                        "irrep_ordering": "GAP Irr(G) / IrreducibleRepresentations(G) row order",
                        "conversion": "Sage complex field to IEEE complex128",
                        "tolerance": 1e-10,
                        "subgroups_included": include_subgroups,
                    },
                }
            )
        ),
        "cayley_table": table,
        "character_real": char_table.real,
        "character_imag": char_table.imag,
        "frobenius_schur": fs,
    }
    for i, cls in enumerate(class_tokens):
        payload[f"class_{i}"] = cls
    for i, matrix in enumerate(irrep_stacks):
        payload[f"irrep_{i}_real"] = matrix.real  # type: ignore[union-attr]
        payload[f"irrep_{i}_imag"] = matrix.imag  # type: ignore[union-attr]
        payload[f"irrep_{i}_character_real"] = char_table[i, class_of].real
        payload[f"irrep_{i}_character_imag"] = char_table[i, class_of].imag
    for i, projector in enumerate(block_projectors):
        payload[f"block_{i}_projector"] = projector
    payload.update(subgroup_payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", type=int, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--include-subgroups",
        action="store_true",
        help=(
            "Also export the full subgroup lattice and every left coset as "
            "separate .npz members. OFF by default: this explodes on "
            "elementary-abelian 2-groups (C2^7 => ~417,000 members) and nothing "
            "consumes coset data yet. Enable only for groups whose coset "
            "templates an instrument needs."
        ),
    )
    args = parser.parse_args()
    default = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "group_artifacts"
        / f"smallgroup_{args.order}_{args.index}.npz"
    )
    print(
        export(
            args.order,
            args.index,
            args.output or default,
            include_subgroups=args.include_subgroups,
        )
    )


if __name__ == "__main__":
    main()
