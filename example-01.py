#!/usr/bin/env python3

# This is a simple example of using MoeaBench
# to run an experiment and visualize the results.
#
# Notes: parameters were set for faster execution in detriment of quality


from MoeaBench import mb

def main():
    
    # 1) Create and configure experiment

    exp = mb.experiment()

    exp.benchmark = mb.benchmarks.DTLZ2()
    exp.moea      = mb.moeas.NSGA3(population=50, generations=50)

    # 2) Run the experiment

    exp.run()
    
    # 3) Visualize results

    mb.spaceplot(exp.front(), mode='static')

    hv = mb.metrics.hypervolume(exp)
    mb.timeplot(hv,mode='static')


if __name__ == "__main__":
    main()
