# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""DTLZ8 research candidate compatibility with production decision geometry."""
import warnings

import numpy as np
import pytest

from moeabench.mops.DTLZ8 import DTLZ8
from scripts.audit.gt_dtlz8_candidate_issue119 import (
    candidate_decisions, candidate_objectives,
)


@pytest.mark.parametrize("m,n", [(2, 21), (3, 32), (4, 43), (5, 52), (10, 103)])
@pytest.mark.parametrize("strategy", ["halton", "lattice", "hybrid"])
def test_nonmultiple_n_preserves_current_evaluation_contract(m, n, strategy):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mop = DTLZ8(M=m, N=n)
    assert any("multiple" in str(w.message) for w in caught)

    decisions = candidate_decisions(
        m, 128, n_variables=n, patch_strategy=strategy
    )
    truth = candidate_objectives(m, 128, patch_strategy=strategy)
    assert decisions.shape == (128, n)
    assert np.all(decisions[:, (n // m) * m:] == 0.5)
    np.testing.assert_allclose(
        mop.evaluation(decisions)["F"], truth, rtol=0, atol=2e-15
    )


@pytest.mark.parametrize("m,n", [(2, 20), (3, 30), (4, 40), (10, 100)])
def test_multiple_n_preserves_decisions_and_objectives(m, n):
    decisions = candidate_decisions(m, 128, n_variables=n,
                                    patch_strategy="hybrid")
    mop = DTLZ8(M=m, N=n)
    np.testing.assert_allclose(
        mop.evaluation(decisions)["F"],
        candidate_objectives(m, 128, patch_strategy="hybrid"),
        rtol=0, atol=2e-15,
    )


def test_invalid_decision_dimension_fails_explicitly():
    with pytest.raises(ValueError, match="at least M"):
        candidate_decisions(4, 128, n_variables=3)
