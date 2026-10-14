# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shared pymoo operator policies used by MoeaBench wrappers."""

from pymoo.operators.mutation.pm import PolynomialMutation


def standard_polynomial_mutation(n_var, eta=20):
    """Return MoeaBench's canonical polynomial mutation operator.

    pymoo >= 0.6 separates the probability of applying mutation to an
    individual (`prob`) from the per-variable mutation probability
    (`prob_var`). MoeaBench's intended policy is the conventional 1/n_var
    probability per variable, so every offspring enters the operator and the
    variable-level gate carries that probability.
    """
    return PolynomialMutation(prob=1.0, prob_var=1.0 / n_var, eta=eta)
