#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

# This is a more complete example of using MoeaBench
# that illustrates how to work with multiple runs.
# 
# Notes: parameters were set for faster execution in detriment of quality


from MoeaBench import mb

def main():
    
    # 1) Create and configure an experiment

    exp1 = mb.experiment()

    exp1.name = "Experiment 1"    # You can name experiments for easier identification.

    exp1.benchmark = mb.benchmarks.DTLZ2()
    exp1.moea      = mb.moeas.NSGA3(population=50, generations=50)
    
    # 2) Run the experiment several times, each time with different random seed.

    exp1.run(5)  # Number of runs set to 5 for faster execution.
    
    # 3) Calculate the hypervolume of each run of the experiment.
    #    This will return a matrix where each row is a generation and 
    #    each column is a run (seed).

    hv1 = mb.hv(exp1)

    # 4) Plot the hypervolume. The plot will show the mean hypervolume of each 
    # run within a shadow of dispersion (standard deviation), along with the 
    # contour of the best and the worst value of the hypervolume.
    
    mb.timeplot(hv1, mode='static', show_bounds=True)

    # 5) Plot the final pareto front.
    #
    # Note: if you write mb.spaceplot(exp1), that will plot the last run 
    # of the experiment --- since exp1 is treated as exp1.front().
    # This design choice highlights the use of exp1.superfront(), which
    # provides the non-dominated solutions considering all runs combined.

    mb.spaceplot(exp1.superfront(), mode='static')

    # 6) If you want to plot all runs independently (to check stability),
    # you can use all_fronts(), which returns a list of individual fronts.
    
    mb.spaceplot(*exp1.all_fronts(), mode='static')



if __name__ == "__main__":
    main()
