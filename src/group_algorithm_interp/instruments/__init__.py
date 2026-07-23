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
* ``endpoints`` -- the Tier-0 endpoint layer (I-01/I-02/I-04/I-06): the per-run
  measurement vector (epochs-to-grok with censoring, unleaked accuracy, the
  chance anchor, the realised-leak covariate) and the paired, estimation-first
  estimators, driven by ``scripts/measure_endpoints.py``.
* ``interventions`` -- the I-07 intervention harness: hook-free zero/mean/
  resample ablation and activation patching of MLP neurons and components, plus
  the shared representational-axis (direction) ablation, all in behavioural
  units against matched controls.
* ``probes`` -- the mechanism arm (I-20/I-26/I-27/I-28, +I-28b): the generic
  functional-form fit, the polycyclic digit probe, the signed-cyclic probe and
  twisted-rule fit, and the power-map / element-order probe, driven by
  ``scripts/measure_probes.py``.
* ``coset`` -- the coset arm and isotypic-block usage for the C2 case study
  (I-15/I-17/I-18/I-19), ``UNDEFINED`` by theorem where no nontrivial core-free
  subgroup exists, driven by ``scripts/measure_coset.py``.
* ``cocycle`` -- the C5 cocycle instrument (I-21/I-22/I-22b/I-22c): extension
  precompute, the ``f == 1`` split discriminator, the model-facing cocycle
  probe and ablation held out over quotient cells, and the twisted-rule FVE fit
  (the extra logit variance the cocycle term explains, held out over quotient
  cells).
* ``audit`` -- the rung-5 circuit audit (I-34), the architecture-confound
  replication (I-36), and the rebuilt shift-invariant direct logit attribution
  (I-35).
* ``nulls`` -- the shared I-03 null battery: the chance anchor and the matched-
  size random-neuron control, collected so one definition serves every caller.
"""
