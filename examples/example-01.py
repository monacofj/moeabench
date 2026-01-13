#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

# This is a simple example of using MoeaBench
# to run an experiment and visualize the results.
#
# Notes: parameters were set for faster execution in detriment of quality


import mb_path
from MoeaBench import mb

def main():
    
    # 1) Create and configure experiment

    exp = mb.experiment()

    exp.mop = mb.mops.DTLZ2()
    exp.moea      = mb.moeas.NSGA3(population=50, generations=50)

    # 2) Run the experiment

    exp.run()
    
    # 3) Visualize results

    mb.spaceplot(exp.front(), mode='static')

    hv = mb.metrics.hypervolume(exp)
    mb.timeplot(hv,mode='static')


if __name__ == "__main__":
    main()
