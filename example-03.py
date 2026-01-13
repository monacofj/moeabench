#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

# This is a more complete example of using MoeaBench
# that compares experiments and the optimization trajectories.
# 
# Notes: parameters were set for faster execution in detriment of quality


from MoeaBench import mb

def main():
    
    # 1) Create and configure two experiments

    exp1 = mb.experiment()
    exp2 = mb.experiment()

    exp1.name = "Experiment 1"      
    exp2.name = "Experiment 2"    

    exp1.benchmark = mb.benchmarks.DTLZ2()
    exp1.moea      = mb.moeas.NSGA3(population=150, generations=5)

    exp2.benchmark = mb.benchmarks.DTLZ2()
    exp2.moea      = mb.moeas.SPEA2(population=100, generations=50)

    # 2) Run the experiments

    exp1.run()
    exp2.run()
    
    # 3) Distinguish dominated and non_dominated solutions (dominance)
    #    Reminder    exp.front() is an alias to exp.non_dominated().objectives
    #    Reminder    exp.non_front() is an alias to exp.dominated().objectives.
 
    mb.spaceplot(exp1.front(), exp1.non_front(), mode='static')

    # 4) Compare two moments of the same experiment (convergence).
    #    exp2 (at gen 50) vs exp2.front(5) (early stage).
 
    mb.spaceplot(exp2.front(5), exp2.front(), mode='static')


   


if __name__ == "__main__":
    main()
