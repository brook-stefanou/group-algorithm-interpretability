"""I-03: the shared null battery.

The suite's null controls were realised piecewise across the instruments that
consume them; this module collects the reusable pieces in one place so a single
definition serves every caller. It is a consolidation, not a redesign -- every
helper reproduces, bit for bit, the control its caller ran before.

The battery has four members, and where each lives:

* **chance anchor** -- the accuracy an untrained model sits at on the
  multiplication task, ``1 / |G|`` (:func:`chance_accuracy`). The endpoint layer
  anchors its accuracy endpoints on it (``endpoints.py`` re-exports this name).
* **random-neuron / permuted-subset control** -- a matched-size random neuron
  set (:func:`random_neuron_control`), the baseline an ablation effect must beat
  to be more than a generic capacity loss. Consumed by the intervention harness
  (``interventions.py``).
* **random-init (untrained-model) null** -- the model training would have started
  from, seeded exactly as the training paths seed it. It is built with the model
  code (``training.trainer.build_model`` after ``seed.set_seed``) rather than
  here, because it needs a live model of the run's own shape; the probes/report
  entry points construct it and pass it in. The occupancy arm's own analytic
  companion is ``occupancy.analytic_null`` (``block_rank_j / |G|``), left where
  it is.
* **random-subspace control** -- a norm/dimension-matched random subspace,
  used by the cocycle ablation (I-22b, ``cocycle.py``), the isotypic-block
  ablation (I-15, ``coset.py``), and the involution-direction ablation (I-28b).
  I-15 and I-22b each keep their own draw so they remain bit-identical to
  their pre-registered form. I-28b's draw is not a separate probes.py
  implementation: it is the same random-direction control the intervention
  harness runs for every ``ablate_direction`` call (``interventions.py``,
  the loop that builds ``random_direction_drop_flips``); ``probes.py`` only
  picks the involution direction and delegates the ablation and its control
  to that harness.

Everything here is deterministic for a fixed seed and never perturbs a global
RNG stream.
"""

from __future__ import annotations

import torch


def chance_accuracy(order: int) -> float:
    """The chance anchor: the accuracy an untrained model sits at on the
    multiplication task, ``1 / |G|`` over the ``|G|`` output classes (I-01's
    untrained-model anchor within the I-03 battery). The grok bar is set well
    above this; reporting it lets a reader see how far above chance an endpoint
    is."""
    if isinstance(order, bool) or not isinstance(order, int):
        raise TypeError(f"group order must be an integer, got {order!r}")
    if order <= 0:
        raise ValueError(f"group order must be positive, got {order}")
    return 1.0 / order


def random_neuron_control(d_mlp: int, n: int, *, seed: int) -> torch.Tensor:
    """A matched-size random neuron set: ``n`` distinct MLP-neuron indices drawn
    from ``range(d_mlp)`` with a private seeded generator, so the draw never
    perturbs the global torch stream. This is the I-03 permuted-neuron / random-
    subset control the intervention harness reads an ablation effect against.

    ``n`` must be within ``[0, d_mlp]``: the matched-size contract is that the
    control set is a subset of the same ``d_mlp`` neurons, and ``randperm``
    would otherwise silently truncate or reinterpret an out-of-range ``n``
    instead of failing."""
    if n < 0 or n > d_mlp:
        raise ValueError(f"n must be between 0 and d_mlp ({d_mlp}), got {n}")
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(d_mlp, generator=generator)[:n]


__all__ = [
    "chance_accuracy",
    "random_neuron_control",
]
