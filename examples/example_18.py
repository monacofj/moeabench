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

from moeabench import mb
import numpy as np

def main():
    mb.system.version()
    # 1. The MaOP Challenge
    # 4-objective DTLZ2 problem. 
    print("\n1. Defining Many-Objective Problem (M=4)...")
    mop = mb.mops.DTLZ2(M=4)
    exp = mb.experiment(mop=mop, moea=mb.moeas.NSGA3(population=40, generations=20))
    exp.run(repeat=1)

    # Attempting a Clinical Report without calibration
    print("\n2. Attempting Clinical Report without calibration...")
    audit_raw = mb.diagnostics.audit(exp)
    print(f"Audit Status: {audit_raw.status.name}")
    print(f"Description: {audit_raw.description}")

    # 2. Local Calibration
    print("\n3. Generating Local Calibration Sidecar...")
    sidecar_path = "dtlz2_m4_demo.json"
    mop.calibrate(
        size=40, 
        source_baseline=sidecar_path,
        force=True
    )
    print(f"Sidecar generated at: {sidecar_path}")

    # 3. Clinical Report with Sidecars
    print("\n4. Final Clinical Report using local sidecar...")
    audit_fixed = mb.diagnostics.audit(exp, source_baseline=sidecar_path)
    audit_fixed.report()

if __name__ == "__main__":
    main()
