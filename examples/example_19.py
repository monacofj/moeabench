#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Example 19: Hypervolume and Ordinal Hypervolume comparison.

This example compares two algorithms on DTLZ2 using two complementary
convergence measures:

* HV preserves metric distances in objective space.
* OHV replaces consecutive distinct reference levels by ordinal unit steps.

Both algorithms are evaluated against the same explicit reference context.
For HV, that context fixes the normalization bounds. For OHV, it fixes the
ordinal axes. Shared references are essential for comparing separate metric
calls fairly.
"""

import mb_path  # noqa: F401  Ensures the repository package is importable.
import moeabench as mb


def main():
    mb.system.version()

    # 1. Configure two independent algorithms on the same problem.
    exp1 = mb.experiment()
    exp1.name = "NSGA-II"
    exp1.mop = mb.mops.DTLZ2(M=3)
    exp1.moea = mb.moeas.NSGA2(
        population=60,
        generations=40,
        seed=7,
    )

    exp2 = mb.experiment()
    exp2.name = "NSGA-III"
    exp2.mop = mb.mops.DTLZ2(M=3)
    exp2.moea = mb.moeas.NSGA3(
        population=60,
        generations=40,
        seed=19,
    )

    # Repeated runs let the plots show the mean trajectory and variability.
    exp1.run(repeat=3)
    exp2.run(repeat=3)

    # 2. Establish one common comparison context.
    #
    # HV derives a shared ideal/nadir normalization box from these final fronts.
    # OHV pools the same final fronts to construct one fixed ordinal lattice.
    reference = [exp1, exp2]

    hv1 = mb.metrics.hv(exp1, ref=reference, mode="exact")
    hv2 = mb.metrics.hv(exp2, ref=reference, mode="exact")
    ohv1 = mb.metrics.ohv(exp1, ref=reference, mode="exact")
    ohv2 = mb.metrics.ohv(exp2, ref=reference, mode="exact")

    # 3. Plot the distance-sensitive and ordinal perspectives separately.
    mb.view.history(
        hv1,
        hv2,
        title="Conventional Hypervolume (HV)",
        show_bounds=True,
    )
    mb.view.history(
        ohv1,
        ohv2,
        title="Ordinal Hypervolume (OHV)",
        show_bounds=True,
    )

    # 4. Reports expose the distinct reference semantics and diagnostics.
    print("\n--- HV: NSGA-II ---")
    hv1.report()
    print("\n--- OHV: NSGA-II ---")
    ohv1.report()


if __name__ == "__main__":
    main()


# Interpretation
# --------------
# A difference visible in HV but attenuated in OHV is primarily associated
# with metric distances between objective values. A difference retained by
# OHV reflects the ordering and coverage of levels in the common ordinal
# reference lattice. Neither metric replaces the other; together they separate
# geometric progress from ordinal progress.
