"""Post-hoc analysis instruments.

Everything here runs offline from a run directory's artifacts (checkpoints,
``run.log``, ``resolved_config.yaml``, ``manifest.yaml``) -- the
analysis-from-snapshots principle: no analysis during training, and every
instrument re-runnable when a criterion changes.

Modules:

* ``checkpoints`` -- dip-aware checkpoint selection (which snapshot an analysis
  reads, with every substitution recorded as data).
* ``occupancy`` -- the spectral core: neuron activations over the Cayley grid
  (I-08), per-neuron concentration (I-09), population irrep occupancy against
  the analytic null ``pi0_j = block_rank_j / |G|`` (I-10), and the
  Dirichlet-derived noise floor with total-variation distances (I-14's
  formulae).
* ``templates`` -- core-free subgroup enumeration (I-12) and the
  ``Ind_H^G 1`` energy-template library (I-13), ``UNDEFINED`` where no
  nontrivial core-free subgroup exists.
* ``report`` -- the record builders the ``scripts/measure_occupancy.py`` entry
  point drives: per-run measurement records with provenance, seed pooling, and
  the I-11 null-calibration gate.
* ``publish`` -- the opt-in W&B publisher: pushes already-measured records to
  W&B as metrics-only runs (never artifacts), gated on ``WANDB_API_KEY``; the
  instrument itself stays fully offline.
"""
