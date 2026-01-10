#!/usr/bin/env python3
from MoeaBench import mb

def main():
    
    exp = mb.experiment()

    exp.benchmark = mb.benchmarks.DTLZ2()
    exp.moea      = mb.moeas.NSGA3()
    
    # Configure for speed
    exp.moea.population = 50
    exp.moea.generations = 50

    exp.run()

    # Access results directly
    print("Objectives:")
    print(exp.objectives())
    
    print("\nVariables:")
    print(exp.variables())

    # print("\nPlotting Pareto Front (Static Mode)...")
    # mb.spaceplot(exp, mode='static')
    
    print("\nPlotting Pareto Front (Static Mode)...")
    mb.spaceplot(exp, mode='static')

if __name__ == "__main__":
    main()
