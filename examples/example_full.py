#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
API Walkthrough
---------------------------------
Two experiments, canonical stats API, and a full tour through
topology, performance, attainment, strata, and clinic diagnostics.
"""

import mb_path
import moeabench as mb
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
        exp1.load(str(exp1_zip))
    else:
        exp1.run(repeat=5)
        exp1.save(str(exp1_zip))

    if exp2_zip.exists():
        exp2.load(str(exp2_zip))
    else:
        exp2.run(repeat=5)
        exp2.save(str(exp2_zip))

    ##
    ## 0) Topology: two fronts + inferred GT
    ##
    mb.view.topology(exp1, exp2)                # Front clouds + inferred GT.
    # Observe overlap with GT (convergence) and cloud spread (diversity).

    ##
    ## 1) Metrics -> report + history/spread/density
    ##
    # Hypervolume: compute, test, and plot with reused stats.

    hv1 = mb.metrics.hv(exp1, scale="abs")  # Falls back to raw if absolute is unavailable.
    hv2 = mb.metrics.hv(exp2, scale="abs")  # Falls back to raw if absolute is unavailable.
    hv1.report()                                 # exp1 HV summary.
    hv2.report()                                 # exp2 HV summary.
    mb.view.history(hv1, hv2)                    # Convergence.

    hv_shift = mb.stats.perf_shift(hv1, hv2)     # MW shift test.
    hv_shift.report()                            # Shift evidence.
    mb.view.spread(hv_shift)                     # Final spread + shift. 

    hv_match = mb.stats.perf_match(hv1, hv2)     # KS match test.
    hv_match.report()                            # Match verdict.
    mb.view.density(hv_match)                    # Density shape + KS verdict.

    hv_win = mb.stats.perf_win(hv1, hv2)         # A12 win prob.
    hv_win.report()                              # Effect size.


    # GD: compute, test, and plot with reused stats.

    gd1 = mb.metrics.gd(exp1)                    # GD trajectory for exp1.
    gd2 = mb.metrics.gd(exp2)                    # GD trajectory for exp2.
    gd1.report()                                 # exp1 GD summary.
    gd2.report()                                 # exp2 GD summary.
    mb.view.history(gd1, gd2)                    # Convergence.

    gd_shift = mb.stats.perf_shift(gd1, gd2)     # MW shift test.
    gd_shift.report()                            # Shift evidence.
    mb.view.spread(gd_shift)                    # Final spread + shift.

    gd_match = mb.stats.perf_match(gd1, gd2)     # KS match test.
    gd_match.report()                            # Match verdict.
    mb.view.density(gd_match)                    # Density shape + KS verdict.

    gd_win = mb.stats.perf_win(gd1, gd2)         # A12 win prob.
    gd_win.report()                              # Effect size.

    # GD+: compute, test, and plot with reused stats.

    gdplus1 = mb.metrics.gdplus(exp1)            # GD+ trajectory for exp1.
    gdplus2 = mb.metrics.gdplus(exp2)            # GD+ trajectory for exp2.
    gdplus1.report()                             # exp1 GD+ summary.
    gdplus2.report()                             # exp2 GD+ summary.
    mb.view.history(gdplus1, gdplus2)            # Convergence.

    gdplus_shift = mb.stats.perf_shift(gdplus1, gdplus2)    # MW shift test.
    gdplus_shift.report()                        # Shift evidence.
    mb.view.spread(gdplus_shift)                 # Final spread + shift.

    gdplus_match = mb.stats.perf_match(gdplus1, gdplus2)    # KS match test.
    gdplus_match.report()                        # Match verdict.
    mb.view.density(gdplus_match)                # Density shape + KS verdict.

    gdplus_win = mb.stats.perf_win(gdplus1, gdplus2)        # A12 win prob.
    gdplus_win.report()                          # Effect size.

    # IGD: compute, test, and plot with reused stats.

    igd1 = mb.metrics.igd(exp1)                  # IGD trajectory for exp1.
    igd2 = mb.metrics.igd(exp2)                  # IGD trajectory for exp2.
    igd1.report()                                # exp1 IGD summary.
    igd2.report()                                # exp2 IGD summary.
    mb.view.history(igd1, igd2)                  # Convergence.

    igd_shift = mb.stats.perf_shift(igd1, igd2)  # MW shift test.
    igd_shift.report()                           # Shift evidence.
    mb.view.spread(igd_shift)                    # Final spread + shift.

    igd_match = mb.stats.perf_match(igd1, igd2)  # KS match test.
    igd_match.report()                           # Match verdict.
    mb.view.density(igd_match)                   # Density shape + KS verdict.

    igd_win = mb.stats.perf_win(igd1, igd2)      # A12 win prob.
    igd_win.report()                             # Effect size.

    # IGD+: compute, test, and plot with reused stats.

    igdplus1 = mb.metrics.igdplus(exp1)          # IGD+ trajectory for exp1.
    igdplus2 = mb.metrics.igdplus(exp2)          # IGD+ trajectory for exp2.
    igdplus1.report()                            # exp1 IGD+ summary.
    igdplus2.report()                            # exp2 IGD+ summary.
    mb.view.history(igdplus1, igdplus2)          # Convergence.

    igdplus_shift = mb.stats.perf_shift(igdplus1, igdplus2)    # MW shift test.
    igdplus_shift.report()                       # Shift evidence.
    mb.view.spread(igdplus_shift)                # Final spread + shift.

    igdplus_match = mb.stats.perf_match(igdplus1, igdplus2)    # KS match test.
    igdplus_match.report()                       # Match verdict.
    mb.view.density(igdplus_match)               # Density shape + KS verdict.

    igdplus_win = mb.stats.perf_win(igdplus1, igdplus2)        # A12 win prob.
    igdplus_win.report()                         # Effect size.

    # EMD: compute, test, and plot with reused stats.

    emd1 = mb.metrics.emd(exp1)                  # EMD trajectory for exp1.
    emd2 = mb.metrics.emd(exp2)                  # EMD trajectory for exp2.
    emd1.report()                                # exp1 EMD summary.
    emd2.report()                                # exp2 EMD summary.
    mb.view.history(emd1, emd2)                  # Convergence.

    emd_shift = mb.stats.perf_shift(emd1, emd2)  # MW shift test.
    emd_shift.report()                           # Shift evidence.
    mb.view.spread(emd_shift)                    # Final spread + shift.

    emd_match = mb.stats.perf_match(emd1, emd2)  # KS match test.
    emd_match.report()                           # Match verdict.
    mb.view.density(emd_match)                   # Density shape + KS verdict.

    emd_win = mb.stats.perf_win(emd1, emd2)      # A12 win prob.
    emd_win.report()                             # Effect size.

    # Front size: compute, test, and plot with reused stats.

    fsize1 = mb.metrics.front_ratio(exp1)        # Front-size trajectory for exp1.
    fsize2 = mb.metrics.front_ratio(exp2)        # Front-size trajectory for exp2.
    fsize1.report()                              # exp1 front-size summary.
    fsize2.report()                              # exp2 front-size summary.
    mb.view.history(fsize1, fsize2)              # Convergence.

    fsize_shift = mb.stats.perf_shift(fsize1, fsize2)  # MW shift test.
    fsize_shift.report()                               # Shift evidence.
    mb.view.spread(fsize_shift)                  # Final spread + shift.

    fsize_match = mb.stats.perf_match(fsize1, fsize2)  # KS match test.
    fsize_match.report()                               # Match verdict.
    mb.view.density(fsize_match)                 # Density shape + KS verdict.

    fsize_win = mb.stats.perf_win(fsize1, fsize2)      # A12 win prob.
    fsize_win.report()                                 # Effect size.
    # Observe trajectory speed (history), final contrast (spread), and tails (density).

    ##
    ## 2) Topology compare aliases
    ##

    # Equivalent to topo_compare(method='ks').
    topo_match = mb.stats.topo_match(exp1, exp2) # KS equivalence test.
    topo_match.report()                          # Objective-space match verdict.

    # Equivalent to topo_compare(method='anderson').
    topo_tail = mb.stats.topo_tail(exp1, exp2)   # Anderson tail test.
    topo_tail.report()                           # Tail-sensitive match verdict.

    # Equivalent to topo_compare(method='emd').
    topo_shift = mb.stats.topo_shift(exp1, exp2, threshold=0.05)  # EMD displacement test.
    topo_shift.report()                                            # Spatial shift diagnosis.

    topo_match_obj = mb.stats.topo_match(exp1, exp2, axes=[0])    # Single-objective axis test.
    topo_match_var = mb.stats.topo_match(exp1, exp2, space="vars", axes=[0])  # Decision-axis test.
    mb.view.density(topo_match_obj)              # Objective-axis density.
    mb.view.density(topo_match_var)              # Decision-axis density.
    # Observe objective-space equivalence versus possible decision-space divergence.

    ##
    ## 3) Attainment and gap
    ##

    att1 = mb.stats.attainment(exp1)                    # Median attainment surface for exp1.
    att2 = mb.stats.attainment(exp2)                    # Median attainment surface for exp2.
    band1_lo = mb.stats.attainment(exp1, level=0.1)     # Lower envelope for exp1.
    band1_hi = mb.stats.attainment(exp1, level=0.9)     # Upper envelope for exp1.
    band2_lo = mb.stats.attainment(exp2, level=0.1)     # Lower envelope for exp2.
    band2_hi = mb.stats.attainment(exp2, level=0.9)     # Upper envelope for exp2.
    gap = mb.stats.attainment_gap(exp1, exp2)           # Localized attainment difference.
    gap.report()                                        # Gap diagnosis.

    mb.view.bands(att1, band1_lo, band1_hi, att2, band2_lo, band2_hi, style="fill")  # Reliability corridor.
    mb.view.topology(att1, att2)                 # Median surfaces in objective space.
    mb.view.gap(gap)                             # Signed local superiority map.
    # Observe corridor width (reliability) and localized superiority regions (gap).

    ##
    ## 4) Strata
    ##

    ranks = mb.stats.ranks(exp1, exp2)           # Rank depth and pressure.
    ranks.report()                               # Rank structure summary.
    mb.view.ranks(ranks)                         # Rank occupancy bars.
 
    strata = mb.stats.strata(exp1, exp2)         # Rank-wise quality distribution.
    strata.report()                              # Strata distribution summary.
    mb.view.strata(strata)                       # Rank quality box summaries.
 
    tiers = mb.stats.tiers(exp1, exp2)           # Shared-tier duel between both groups.
    tiers.report()                               # Tier duel summary.
    mb.view.tiers(tiers)                         # Shared-tier stacked duel.
    # Observe selection pressure depth and class occupancy profile.

    ##
    ## 5) Clinic audit
    ##

    diag1 = mb.clinic.audit(exp1)                # Clinical synthesis for exp1.
    diag2 = mb.clinic.audit(exp2)                # Clinical synthesis for exp2.
    diag1.report(full=True)                      # Full audit narrative.
    diag2.report(full=True)                      # Full audit narrative.

    mb.view.radar(diag1, diag2)                  # Global health radar.

    close1 = mb.clinic.closeness(exp1)           # Closeness pathology for exp1.
    close2 = mb.clinic.closeness(exp2)           # Closeness pathology for exp2.
    close1.report()                              # Closeness summary.
    close2.report()                              # Closeness summary.
    mb.view.ecdf(close1)                         # Quantile profile.
    mb.view.ecdf(close2)                         # Quantile profile.
    mb.view.density(close1)                      # Distribution morphology.
    mb.view.density(close2)                      # Distribution morphology.
    mb.view.history(close1)                      # Temporal pathology evolution.
    mb.view.history(close2)                      # Temporal pathology evolution.
    # Observe global health shape (radar), quantiles (ecdf), and pathology morphology over time.


if __name__ == "__main__":
    main()
