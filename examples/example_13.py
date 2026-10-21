#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Example 13: Reference-based Quality Validation (Q-Scores)
=========================================================

Evaluate the quality of an NSGA-II approximation to the DTLZ1 Pareto front.
The Q-scores summarize different aspects of approximation quality, while
clinical markers show the closeness of individual plotted solutions.

Both the audit and the visualization use the same optimization run and the
same calibrated reference front. Their interpretation depends on the
measured results, rather than assuming a particular pathology beforehand.
"""

import mb_path
import moeabench as mb


def main():
    mb.system.version()

    # 1. Run one reproducible NSGA-II experiment on three-objective DTLZ1.
    exp = mb.experiment()
    exp.mop = mb.mops.DTLZ1(M=3)
    exp.moea = mb.moeas.NSGA2(population=100, generations=280)
    exp.name = "NSGA-II on DTLZ1"
    exp.run(seed=1)

    # 2. Evaluate against the registered calibration GT. The Q-score
    # baselines were produced for this precise reference; passing an unrelated
    # 500-point analytical sampling could invalidate the clinical scores.
    audit = mb.clinic.audit(exp)
    if audit.quality is None:
        raise RuntimeError(
            "Example 13 requires compatible FAIR calibration baselines."
        )
    audit.quality.report(full=True)

    # 3. Draw exactly the reference used by the audit, not a new sampling.
    # The gt keyword adds one reference trace; no second run is needed.
    gt = audit.diagnostic_context["GT"]
    mb.view.topology(exp, gt=gt, markers=True)


if __name__ == "__main__":
    main()
