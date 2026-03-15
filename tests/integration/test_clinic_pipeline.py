# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import moeabench as mb


def test_clinic_pipeline(paired_experiments, canonical_gt):
    exp1, exp2 = paired_experiments

    diag1 = mb.clinic.audit(exp1)
    diag2 = mb.clinic.audit(exp2)
    close1 = mb.clinic.closeness(exp1, ref=canonical_gt)
    close2 = mb.clinic.closeness(exp2, ref=canonical_gt)

    assert diag1.q_audit_res is not None
    assert diag1.fair_audit_res is not None
    assert diag2.q_audit_res is not None
    assert close1.history_values is not None
    assert close2.raw_data is not None

    info = mb.system.info(show=False)
    assert "python_version" in info

    assert mb.view.radar(diag1, diag2, mode="static", show=False) is not None
    assert mb.view.ecdf(close1, mode="static", show=False) is not None
    assert mb.view.ecdf(close2, mode="static", show=False) is not None
    assert mb.view.density(close1, mode="static", show=False) is not None
    assert mb.view.history(close1, mode="static", show=False) is not None
