#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Example 18: Clinical Report & MaOP Calibration
----------------------------------------------
This tutorial demonstrates how to handle Many-Objective Optimization Problems (MaOP)
where clinical baselines are not pre-calculated in the library registry.
"""

import moeabench as mb
import numpy as np

def main():
    mb.system.version()
    # 1. The MaOP Challenge
    # 4-objective DTLZ2 problem. 
    mop = mb.mops.DTLZ2(M=4)
    exp = mb.experiment(mop=mop, moea=mb.moeas.NSGA3(population=40, generations=20))
    exp.run(repeat=1)

    # Attempting a Clinical Report without calibration
    audit_raw = mb.clinic.audit(exp)

    # 2. Local Calibration
    sidecar_path = "dtlz2_m4_demo.json"
    mop.calibrate(
        size=40, 
        source_baseline=sidecar_path,
        force=True
    )

    # 3. Clinical Report with Sidecars
    audit_fixed = mb.clinic.audit(exp, source_baseline=sidecar_path)
    audit_fixed.report()

if __name__ == "__main__":
    main()
