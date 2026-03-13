#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Example 06: Statistical Hypothesis Testing
------------------------------------------
This example demonstrates how to perform rigorous statistical comparisons 
between two algorithms using non-parametric tests and effect size measures.
"""

import mb_path
import moeabench as mb

def main():
    mb.system.version()
    
    # 1. Setup: Compare NSGA-III and SPEA2 with 10 repetitions
    repeats = 10
    pop_size = 100
    gens = 50
    
    exp1 = mb.experiment()
    exp1.name = "NSGA-III"
    exp1.mop = mb.mops.DTLZ2(M=3)
    exp1.moea = mb.moeas.NSGA3(population=pop_size, generations=gens)

    exp2 = mb.experiment()
    exp2.name = "SPEA2"
    exp2.mop = mb.mops.DTLZ2(M=3)
    exp2.moea = mb.moeas.SPEA2(population=pop_size, generations=gens)

    exp1.run(repeat=repeats)
    exp2.run(repeat=repeats)

    # 2. Statistical Inference
    
    # [mb.stats.perf_shift] uses Mann-Whitney U
    # Equivalent to mb.stats.perf_compare(method='mannwhitney').
    # to check for significant location differences.
    res1 = mb.stats.perf_shift(exp1, exp2, metric=mb.metrics.hv)
    res1.report()
    
    # [mb.stats.perf_match] uses Kolmogorov-Smirnov (KS)
    # Equivalent to mb.stats.perf_compare(method='ks').
    # to compare distribution shapes.
    res2 = mb.stats.perf_match(exp1, exp2, metric=mb.metrics.hv)
    res2.report()
    
    # [mb.stats.perf_win] uses Vargha-Delaney A12
    # Equivalent to mb.stats.perf_compare(method='a12').
    # to estimate win probability / effect size.
    res3 = mb.stats.perf_win(exp1, exp2, metric=mb.metrics.hv)
    res3.report()

if __name__ == "__main__":
    main()

# Statistical tests avoid the trap of "visual optimization." 
# The Mann-Whitney U test tells us if one algorithm is significantly better 
# than the other based on the median performance.
#
# In moeabench, these tests are highly integrated: they automatically handle the 
# extraction of metrics (like Hypervolume) and set a common reference point.
