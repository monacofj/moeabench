#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

# This is a more complete example of using MoeaBench
# that compares two different experiments.
# 
# Notes: parameters were set for faster execution in detriment of quality


import mb_path
from MoeaBench import mb

def main():
    
    # 1) Create and configure two experiments

    exp1 = mb.experiment()
    exp2 = mb.experiment()

    exp1.name = "Experiment 1"    # You can name experiments for easier  
    exp2.name = "Experiment 2"    # identification in the plots.

    exp1.mop = mb.mops.DTLZ2()
    exp1.moea      = mb.moeas.NSGA3(population=50, generations=50)

    exp2.mop = mb.mops.DTLZ2()
    exp2.moea      = mb.moeas.SPEA2(population=50, generations=50)

    # 2) Run the experiments

    exp1.run()
    exp2.run()
    
    # 3) Space plot. Both experiments will be plotted in the same figure.
 
    mb.spaceplot(exp1, exp2, mode='static')

    # 4) Time plot. Both experiments will be plotted in the same figure.
    #    For a fair comparision, you can set the scale reference.

    ref = [exp1, exp2] 

    hv1 = mb.metrics.hypervolume(exp1, ref)  # Both metrics will be scaled
    hv2 = mb.metrics.hypervolume(exp2, ref)  # to the same reference.
    
    mb.timeplot(hv1, hv2, mode='static')


if __name__ == "__main__":
    main()
