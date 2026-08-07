#!/usr/bin/env python3
"""Run the existing probe bootstrap with an explicit random probe initialization.

The paper probe uses scikit-learn's default L-BFGS solver. Its ``random_state``
argument does not itself randomize the starting coefficients. This rebuttal-only
wrapper retains the same logistic-regression settings and bootstrap evaluator,
but supplies seeded random coefficients to L-BFGS through ``warm_start``.
"""

from __future__ import annotations

import os

import numpy as np
from sklearn.linear_model import LogisticRegression

from scripts.ablations import evaluate_linear_probe_bootstrap as bootstrap


class RandomlyInitializedLogisticRegression(LogisticRegression):
    """Logistic regression with a reproducible random L-BFGS starting point."""

    def fit(self, X, y, sample_weight=None):
        initialization_seed = (
            0 if self.random_state is None else int(self.random_state)
        )
        configured_scale = os.environ.get("LINEAR_PROBE_INIT_SCALE")
        initialization_scale = (
            float(configured_scale)
            if configured_scale is not None
            else float(np.sqrt(2.0 / (X.shape[1] + 1)))
        )
        rng = np.random.default_rng(initialization_seed)

        # For a binary probe, scikit-learn stores one coefficient vector.
        self.warm_start = True
        self.coef_ = rng.normal(
            loc=0.0,
            scale=initialization_scale,
            size=(1, X.shape[1]),
        )
        self.intercept_ = rng.normal(
            loc=0.0,
            scale=initialization_scale,
            size=(1,),
        )
        print(
            "[probe-init] "
            f"seed={initialization_seed} scale={initialization_scale:.8g}"
        )
        return super().fit(X, y, sample_weight=sample_weight)


def main() -> None:
    # Reuse the original feature extraction, stratified bootstrap, prediction,
    # output, and completion logic without modifying the paper/base script.
    bootstrap.LogisticRegression = RandomlyInitializedLogisticRegression
    bootstrap.main()


if __name__ == "__main__":
    main()
