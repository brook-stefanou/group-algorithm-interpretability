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
* **random-subspace control** -- a norm/dimension-matched random subspace, used
  by the cocycle (I-22b) and involution-direction (I-28b) ablations; those keep
  their own draws so the two remain bit-identical to their pre-registered form.

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
    if order <= 0:
        raise ValueError(f"group order must be positive, got {order}")
    return 1.0 / order


def random_neuron_control(d_mlp: int, n: int, *, seed: int) -> torch.Tensor:
    """A matched-size random neuron set: ``n`` distinct MLP-neuron indices drawn
    from ``range(d_mlp)`` with a private seeded generator, so the draw never
    perturbs the global torch stream. This is the I-03 permuted-neuron / random-
    subset control the intervention harness reads an ablation effect against."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(d_mlp, generator=generator)[:n]


__all__ = [
    "chance_accuracy",
    "random_neuron_control",
]
