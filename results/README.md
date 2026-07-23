# `results/` — committed derivation outputs

Reviewer-facing artefacts produced by a script under [`scripts/`](../scripts/),
committed so a result can be cited and inspected without re-running the
(GAP-dependent) derivation. Each file records the exact input hashes it was
produced from; re-running the producing script against the same pinned inputs
reproduces it byte-for-byte.

## `falsifier_screen_results_full.json`

The mechanically-derived Fourier-falsifier screen over every group in
`data/group_properties_full.jsonl` with 21 ≤ order ≤ 255 (6,958 groups).
`genuine_clean_pairs` (370 pairs across 450 distinct groups) is the
authoritative, complete clean-pair family that **defines test C1**; the core
study's selection rule is applied to that list.

- Produced by: `scripts/derive_falsifiers.py` (drives `scripts/falsifier_lib.g`
  in GAP). Re-run: `uv run python scripts/derive_falsifiers.py`.
- Input: `data/group_properties_full.jsonl`
  sha256 `b3071828c7d13eceb90453ad24c87e89adf402784f4a3c11d1d77e7e34b0536b`.
- sha256 of this file:
  `2cf5099e698fca4ac54d100d6e376b46b46c275c2f6fe9c0b6ace82519bf9722`.

## `falsifier_panel_rows.json`

The pinned 363-row `(order, index)` panel (provenance: the parsed run
register). Used as the reference panel for the results file's
`consistency_check_vs_panel` cross-check, and as the default `--panel` input for
the panel-restricted variant of the screen
(`scripts/derive_falsifiers.py --panel results/falsifier_panel_rows.json`).

- sha256 of this file:
  `bc4617e7afcd1fb6db98df2f7cbc57655c519dbed1bd47f72e8b9017e870a1ec`.
