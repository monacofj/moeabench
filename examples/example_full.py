#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Example Full: New API Walkthrough
---------------------------------
Two experiments, full analysis tour.
Order follows `misc/new-api.md` mapping.
"""

import mb_path
from moeabench import mb
from pathlib import Path


def main():
    mb.system.version()
    ##
    ## Setup: same MOP, two MOEAs
    ##
    mop = mb.mops.DTLZ2(M=3)
    #mop.calibrate()  # Ensure baseline/GT is available for this M (e.g., M=4).
    exp1 = mb.experiment()
    exp1.name = "NSGA-II"
    exp1.mop = mop
    exp1.moea = mb.moeas.NSGA2(population=100, generations=80)

    exp2 = mb.experiment()
    exp2.name = "NSGA-III"
    exp2.mop = mop
    exp2.moea = mb.moeas.NSGA3(population=100, generations=80)

    exp1_zip = Path(__file__).with_name("example_full_exp1.zip")
    exp2_zip = Path(__file__).with_name("example_full_exp2.zip")

    if exp1_zip.exists():
        exp1.load(str(exp1_zip), mode="all")
    else:
        exp1.run(repeat=5)
        exp1.save(str(exp1_zip), mode="all")

    if exp2_zip.exists():
        exp2.load(str(exp2_zip), mode="all")
    else:
        exp2.run(repeat=5)
        exp2.save(str(exp2_zip), mode="all")

    ##
    ## 0) Topology: two fronts + inferred GT
    ##
    mb.view.topology(exp1, exp2)
    # Observe overlap with GT (convergence) and cloud spread (diversity).

    ##
    ## 1) Metrics -> report + history/spread/density
    ##
    gt = mop.pf(n_points=3000)

    # Hypervolume: compute, test, and plot with reused stats.

    hv1 = mb.metrics.hv(exp1, scale="absolute")  # Falls back to raw if absolute is unavailable.
    hv2 = mb.metrics.hv(exp2, scale="absolute")  # Falls back to raw if absolute is unavailable.
    hv1.report()                                 # exp1 HV summary.
    hv2.report()                                 # exp2 HV summary.

    hv_shift = mb.stats.perf_shift(hv1, hv2)     # MW shift test.
    hv_match = mb.stats.perf_match(hv1, hv2)     # KS match test.
    hv_win = mb.stats.perf_win(hv1, hv2)         # A12 win prob.
    hv_shift.report()                            # Shift evidence.
    hv_match.report()                            # Match verdict.
    hv_win.report()                              # Effect size.

    mb.view.history(hv1, hv2)                    # Convergence.
    mb.view.spread(hv1, hv2, stats=hv_shift)     # Final spread + shift.
    mb.view.density(hv1, hv2, stats=hv_match)    # Density shape + KS verdict.

    # GD: compute, test, and plot with reused stats.

    gd1 = mb.metrics.gd(exp1, ref=gt)            # GD trajectory for exp1.
    gd2 = mb.metrics.gd(exp2, ref=gt)            # GD trajectory for exp2.
    gd1.report()                                 # exp1 GD summary.
    gd2.report()                                 # exp2 GD summary.

    gd_shift = mb.stats.perf_shift(gd1, gd2)     # MW shift test.
    gd_match = mb.stats.perf_match(gd1, gd2)     # KS match test.
    gd_win = mb.stats.perf_win(gd1, gd2)         # A12 win prob.
    gd_shift.report()                            # Shift evidence.
    gd_match.report()                            # Match verdict.
    gd_win.report()                              # Effect size.

    mb.view.history(gd1, gd2)                                  # Convergence.
    mb.view.spread(gd1, gd2, stats=gd_shift)     # Final spread + shift.
    mb.view.density(gd1, gd2, stats=gd_match)    # Density shape + KS verdict.

    # GD+: compute, test, and plot with reused stats.

    gdplus1 = mb.metrics.gdplus(exp1, ref=gt)    # GD+ trajectory for exp1.
    gdplus2 = mb.metrics.gdplus(exp2, ref=gt)    # GD+ trajectory for exp2.
    gdplus1.report()                             # exp1 GD+ summary.
    gdplus2.report()                             # exp2 GD+ summary.

    gdplus_shift = mb.stats.perf_shift(gdplus1, gdplus2)    # MW shift test.
    gdplus_match = mb.stats.perf_match(gdplus1, gdplus2)    # KS match test.
    gdplus_win = mb.stats.perf_win(gdplus1, gdplus2)        # A12 win prob.
    gdplus_shift.report()                        # Shift evidence.
    gdplus_match.report()                        # Match verdict.
    gdplus_win.report()                          # Effect size.

    mb.view.history(gdplus1, gdplus2)                       # Convergence.
    mb.view.spread(gdplus1, gdplus2, stats=gdplus_shift)    # Final spread + shift.
    mb.view.density(gdplus1, gdplus2, stats=gdplus_match)   # Density shape + KS verdict.

    # IGD: compute, test, and plot with reused stats.

    igd1 = mb.metrics.igd(exp1, ref=gt)          # IGD trajectory for exp1.
    igd2 = mb.metrics.igd(exp2, ref=gt)          # IGD trajectory for exp2.
    igd1.report()                                # exp1 IGD summary.
    igd2.report()                                # exp2 IGD summary.

    igd_shift = mb.stats.perf_shift(igd1, igd2)  # MW shift test.
    igd_match = mb.stats.perf_match(igd1, igd2)  # KS match test.
    igd_win = mb.stats.perf_win(igd1, igd2)      # A12 win prob.
    igd_shift.report()                           # Shift evidence.
    igd_match.report()                           # Match verdict.
    igd_win.report()                             # Effect size.

    mb.view.history(igd1, igd2)                  # Convergence.
    mb.view.spread(igd1, igd2, stats=igd_shift)  # Final spread + shift.
    mb.view.density(igd1, igd2, stats=igd_match) # Density shape + KS verdict.

    # IGD+: compute, test, and plot with reused stats.

    igdplus1 = mb.metrics.igdplus(exp1, ref=gt)  # IGD+ trajectory for exp1.
    igdplus2 = mb.metrics.igdplus(exp2, ref=gt)  # IGD+ trajectory for exp2.
    igdplus1.report()                            # exp1 IGD+ summary.
    igdplus2.report()                            # exp2 IGD+ summary.

    igdplus_shift = mb.stats.perf_shift(igdplus1, igdplus2)    # MW shift test.
    igdplus_match = mb.stats.perf_match(igdplus1, igdplus2)    # KS match test.
    igdplus_win = mb.stats.perf_win(igdplus1, igdplus2)        # A12 win prob.
    igdplus_shift.report()                       # Shift evidence.
    igdplus_match.report()                       # Match verdict.
    igdplus_win.report()                         # Effect size.

    mb.view.history(igdplus1, igdplus2)                        # Convergence.
    mb.view.spread(igdplus1, igdplus2, stats=igdplus_shift)    # Final spread + shift.
    mb.view.density(igdplus1, igdplus2, stats=igdplus_match)   # Density shape + KS verdict.

    # EMD: compute, test, and plot with reused stats.

    emd1 = mb.metrics.emd(exp1, ref=gt)          # EMD trajectory for exp1.
    emd2 = mb.metrics.emd(exp2, ref=gt)          # EMD trajectory for exp2.
    emd1.report()                                # exp1 EMD summary.
    emd2.report()                                # exp2 EMD summary.

    emd_shift = mb.stats.perf_shift(emd1, emd2)  # MW shift test.
    emd_match = mb.stats.perf_match(emd1, emd2)  # KS match test.
    emd_win = mb.stats.perf_win(emd1, emd2)      # A12 win prob.
    emd_shift.report()                           # Shift evidence.
    emd_match.report()                           # Match verdict.
    emd_win.report()                             # Effect size.

    mb.view.history(emd1, emd2)                  # Convergence.
    mb.view.spread(emd1, emd2, stats=emd_shift)  # Final spread + shift.
    mb.view.density(emd1, emd2, stats=emd_match) # Density shape + KS verdict.

    # Front size: compute, test, and plot with reused stats.

    fsize1 = mb.metrics.front_ratio(exp1)        # Front-size trajectory for exp1.
    fsize2 = mb.metrics.front_ratio(exp2)        # Front-size trajectory for exp2.
    fsize1.report()                              # exp1 front-size summary.
    fsize2.report()                              # exp2 front-size summary.

    fsize_shift = mb.stats.perf_shift(fsize1, fsize2)  # MW shift test.
    fsize_match = mb.stats.perf_match(fsize1, fsize2)  # KS match test.
    fsize_win = mb.stats.perf_win(fsize1, fsize2)      # A12 win prob.
    fsize_shift.report()                               # Shift evidence.
    fsize_match.report()                               # Match verdict.
    fsize_win.report()                                 # Effect size.

    mb.view.history(fsize1, fsize2)                           # Convergence.
    mb.view.spread(fsize1, fsize2, stats=fsize_shift)         # Final spread + shift.
    mb.view.density(fsize1, fsize2, stats=fsize_match)        # Density shape + KS verdict.
    # Observe trajectory speed (history), final contrast (spread), and tails (density).

    ##
    ## 2) Topology compare aliases
    ##

    # Equivalent to topo_compare(method='ks').
    topo_match = mb.stats.topo_match(exp1, exp2)
    topo_match.report()

    # Equivalent to topo_compare(method='anderson').
    topo_tail = mb.stats.topo_tail(exp1, exp2)
    topo_tail.report()

    # Equivalent to topo_compare(method='emd').
    topo_shift = mb.stats.topo_shift(exp1, exp2, threshold=0.05)
    topo_shift.report()

    mb.view.density(exp1, exp2, axes=[0])
    mb.view.density(exp1, exp2, space="vars", axes=[0])
    # Observe objective-space equivalence versus possible decision-space divergence.

    ##
    ## 3) Attainment and gap
    ##

    att1 = mb.stats.attainment(exp1, level=0.5)
    att2 = mb.stats.attainment(exp2, level=0.5)
    gap = mb.stats.attainment_gap(exp1, exp2, level=0.5)
    gap.report()

    mb.view.bands(exp1, exp2, levels=[0.1, 0.5, 0.9])
    mb.view.topology(att1, att2)
    mb.view.gap(exp1, exp2)
    # Observe corridor width (reliability) and localized superiority regions (gap).

    ##
    ## 4) Strata
    ##

    strata1 = mb.stats.strata(exp1)
    strata2 = mb.stats.strata(exp2)
    strata1.report()
    strata2.report()

    mb.view.ranks(strata1, strata2)
    mb.view.caste(strata1, strata2)
    mb.view.tiers(exp1, exp2)
    # Observe selection pressure depth and class occupancy profile.

    ##
    ## 5) Clinic audit
    ##

    diag1 = mb.clinic.audit(exp1, quality=True)
    diag2 = mb.clinic.audit(exp2, quality=True)
    diag1.report(full=True)
    diag2.report(full=True)

    mb.view.radar(diag1)
    mb.view.radar(diag2)
    mb.view.ecdf(exp1, ground_truth=gt, metric="closeness")
    mb.view.ecdf(exp2, ground_truth=gt, metric="closeness")
    mb.view.density(exp1, ground_truth=gt, metric="closeness")
    mb.view.density(exp2, ground_truth=gt, metric="closeness")
    mb.view.history(exp1, ground_truth=gt, metric="closeness")
    mb.view.history(exp2, ground_truth=gt, metric="closeness")
    # Observe global health shape (radar), quantiles (ecdf), and pathology morphology over time.


if __name__ == "__main__":
    main()
