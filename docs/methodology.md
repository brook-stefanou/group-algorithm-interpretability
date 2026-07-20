# Methodology

How the study is designed and how its results get reported. The core study —
the groups it trains, the five claims it makes, and the instruments behind
each claim — is specified in [core-study.md](core-study.md); the directions
it could grow in are in [extensions.md](extensions.md). What follows are the
commitments that hold across both.

## The task and the question

Networks are trained on finite-group multiplication: given a pair of group
elements `(a, b)`, predict the product `a*b`, from a partial multiplication
table with a held-out split. The question is what algorithm a trained network
implements and how that depends on the group. The existing circuit-level
accounts — a representation-theoretic (Fourier) account, a coset account, and
convergence theorems proved outside this training regime — make different
predictions on particular groups, and some forbid any difference within
particular pairs. I train on contrasts chosen so those predictions can be
checked.

## Group selection

Group selection is manual and pre-registered: a fixed, written list with a
stated rationale per inclusion, recorded before the runs it justifies. Nothing
in this repository selects groups automatically. Groups are identified by
their canonical SmallGroups `(order, index)` pair; names and family labels
(`D32`, "extraspecial") are descriptive shorthand. Every invariant used to
reason about a candidate is exported from GAP/SageMath
(`scripts/enumerate_groups.g`, `scripts/enumerate_groups.py`,
`scripts/export_group.py`); the Python training code recomputes none of it.

A group earns a slot because it makes an algorithmic route available,
unavailable, or differently costly — a Fourier route, a coset action, a carry
chain, an extension cocycle — and because that difference separates competing
accounts of what the network computes.

The headline family (claim C1) is criterion-defined. A re-runnable screen
(`scripts/derive_falsifiers.py`, output committed at
`results/falsifier_screen_results_full.json`) over the committed invariant
dataset (`data/group_properties_full.jsonl`) finds pairs of groups with
identical character tables, identical Frobenius–Schur indicators, and no
core-free subgroup at any index whose induced-representation template separates
the pair — pairs on which the Fourier account and the coset account both
predict no within-pair difference. The selection rule is closed:
the five pre-registered panel pairs plus every screen-clean pair at order
≤ 81, eleven pairs in total. The full list, each pair's role, and the staging
rule that gates the five heaviest pairs behind the width-256 probe cells are
in [core-study.md](core-study.md); the decisions that produced this design
are dated in the [research log](research-log.md).

## Reporting

A run's record carries the measurements taken from that trained model — a
signal vector with provenance (run id, checkpoint, git revision, config hash,
data hash) — and no threshold inside a run turns those measurements into a
label naming the algorithm the model learned. Cross-group comparison,
clustering, and any typology are computed from those records afterwards, so a
typology can be revised or abandoned without invalidating the runs it was
drawn from.

Reporting is estimation-first. Every endpoint is an effect size with a
confidence interval, and every measurement taken is reported. There's no
significance threshold, no smallest-effect-size-of-interest, and no
multiple-comparison gate deciding what ships. Descriptive p-values (paired
sign-flip permutation tests) may accompany an interval as a supplement, never
as a gate. A narrow interval around zero is evidence against a large effect;
a wide interval is uninformative; both readings come from the interval itself.

The falsification contrasts inherit this. Where an account forbids a
within-pair difference — identical character tables force the Fourier account
to predict identical behaviour on both members — the paired interval on that
pair is the test. A reproducible nonzero difference is evidence against the
account; a tight interval around zero is the account surviving its sharpest
available behavioural test, reported as an interval upper bound.

## Paired design

Seeds are paired: the same 50 seeds on both members of every pair, with
`data.split_seed` pinned across the whole campaign so every run sees the same
train/test split and only initialisation varies. An unmatched seed is dropped
so the pairing is never broken. Differences are aggregated within pair first;
the unit of any cross-group statement is the pair, and confidence intervals
come from a paired bootstrap over seeds within a pair. The C1 pairs are
reported pair by pair — one estimate and one interval per pair, seeds never
pooled across pairs. Nine are replication across primes and orders; the other two are
direct-product lifts of the `(27,3)/(27,4)` contrast and are read as
persistence-under-embedding probes of that pair.

## Endpoints

1. Epochs-to-grok: the first epoch beginning a streak of five or more
   consecutive epochs above 0.99 accuracy on the transpose-unleaked held-out
   subset. The training ceiling is 60,000 epochs
   (`configs/experiment/core.yaml`), pre-registered and hard: a seed that has
   not grokked by then is recorded as censored (>60k), with no extension and
   no resume. One-side-censored pairs keep their sign;
   both-censored pairs are ties; the censoring fraction is itself a reported
   per-group measurement.
2. Accuracy means transpose-unleaked held-out accuracy everywhere. A test pair
   `(a, b)` is leaked when its transpose `(b, a)` is in the training set and
   the two elements commute, so the label is readable off a memorised training
   example. The realised leak fraction is recorded per run as a covariate, and
   raw test accuracy is never used for a cross-group number.
3. Mechanism endpoints — irrep-occupancy vectors measured against each group's
   analytic null, and probe, ablation, and functional-form-fit outputs — are
   specified per claim in [core-study.md](core-study.md), each with its own
   null and its effect size in behavioural units.

## Analysis from snapshots

No analysis runs during training. A run writes trajectory snapshots (dense
early, then at a fixed interval), and every instrument and every transition
criterion runs post-hoc from those snapshots, so an analysis criterion can
change without retraining anything. Analysis scripts are frozen and hashed
before the first analysis is read. The first instrument is built:
`scripts/measure_occupancy.py` measures irrep occupancy from a run directory
offline, selecting each run's checkpoint through the dip-aware rule in
[the reproducibility contract](reproducibility.md).

## Scope

- One architecture: a one-layer transformer at width 128, with named cells
  duplicated at width 256 (the cell list in `configs/campaign/core.yaml`
  fixes each cell's width). Every result is conditional on that regime. A
  fully connected replication is scoped to the C2 case study.
- Cross-group contrasts are associations. No intervention inside one model can
  test a claim about the difference between two models; only within-model
  mechanism claims (the D32 circuit account, the carry ablation, the cocycle
  probe) go beyond association.
- A clean-pair difference falsifies without identifying a cause: the pairs
  still differ on element-order structure, automorphism-group order, and
  subgroup counts, so no single residual invariant is singled out.
- Absolute difficulty numbers on abelian groups are never compared across
  groups — the transpose leak inflates them. Only the within-pair contrast is
  claimed.
- Grokking timing is reported separately from the inferred algorithm. Dynamics
  are evidence about training; mechanism claims rest on the probe, ablation,
  and fit instruments.
