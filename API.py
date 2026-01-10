
## New API
## This script is for documentation purposes only. Do not execute it.

# Instantiating and configuring an experiment

exp = mb.experiment()

exp.benchmark = mb.benchmarks.DTLZ2()
exp.moea      = mb.moeas.NSGA3()
    
# Running the experiment

exp.run(30)              # Execute 30 runs and save all them

# Result types (experiments)

exp				         # MultiExperiment object (contain all 30 runs)

exp[4]				     # Experiment object (contain only the 5th run)

# Canonical result selector: where (run), when (generation), what (objectives or variables)


exp[4].pop(100).objectives  # Objs of 100th gen, of the 5th run.

exp[4].pop(100).variables   # Vars of 100th gen, of the 5th run.

exp[4].pop().objectives     # Objs of last gen, of the 5th run.

exp.pop(100).objectives     # Objs of 100th gen, or all run.

exp.pop().objectives         # Objs of last gen, or all run.

exp.objectives               # Like exp.pop().objectives

# Shortcuts

exp.last_run                 # exp[-1]

exp.last_run.last_pop        # exp[-1].pop()

exp.last_pop                 # exp[-1].pop()

# Filters

exp[4].pop()                   # All population 4th run, last gen.

exp[4].nondominated()          # Only the non-dominated (run 4, last gen)

exp[4].dominated()             # Only the dominated (run 4, last gen)

exp[4].front()                 # Same as exp[4].nondominated().objectives

exp[4].set()                   # Same as exp[4].nondominated().variables

# Aliases

exp[4].pop().objs              # Alias to exp[4].pop().objectives

exp[4].pop().vars             # Alias to exp[4].pop().variables 

# If one one run (it follows naturally)

exp.run()              # One single run

exp.front()            # Front of the last run (the unique run)

exp.set()              # Set of the last run (the unique run)

# Note: pop(), dominated(), nondominated() take gen as argument.

# Metrics

hv = mb.metrics.hypervolume(exp)    # Returns a matrix of hypervolumes:
                                    # lines are gens, columns are runs

hv # Matrix NumPy of GxR

hv.runs[i] # Selector of the trajectory, alls gens in i-th run

hv.gens[i] # Selector of the statistic distribution, i-th gen

# Global comparison (reference scales)

ref = [exp1, exp2]
hypervolume(exp1, ref=ref)  # We ref to compute the hypervolume of exp1

# The same holds for the other metrics implemented by moeabench

# Plots (works for hypervolume, igd, etc.)

mb.timeplot(hv)           # Plot the hv.gens (mean and dispersion shadow)
mb.timeplot(hv1, hv2)     # Plot the hv.gens of two experiments

mb.spaceplot(exp.front())   # Plot the front of the last run
mb.spaceplot(exp.front(), exp2.front())   # Plot the front of two experiments
mb.spaceplot(exp)       #
mb.spaceplot(exp.front(100), exp.front()) # Compare two moments

# In the above examples, if the argument is exp use its name (if no name, the implicit name is the object name itselfOtherwise, get from argument label. Think of other usages too.ArithmeticError

mb.spaceplot(exp.nondominated(), exp.front())



# Save experiment.
# Save an experiment in a folder and load it back in another program.
# MoeaBench will save the benchmark problem, the optimization algorithm,and
# the optimization results if any.

exp.save("my_experiment")  # Save the experiment as my_experiment.mb

exp4.load("my_experiment") # Load the my_experiment.mb
exp4.run()                  # and run it again, if desired.

# Moeabench has a built-in set of moeas and benchmarks. Perhaps they needed to be adapted to the new API.

# Moeabench can be extended via add_benchmark and add_moea. Perhaps they need to be updated to the new API.





