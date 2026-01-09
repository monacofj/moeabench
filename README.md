# MoeaBench

MoeaBench is a Python framework designed for running benchmarks for Multiobjective Evolutionary Algorithms (MOEA).

The package offers facilities to programmatically create benchmark programs and extract standard performance measurements. Additionally, it provides graphical capabilities to produce time-plots of algorithm convergence and 3D visualizations of Pareto fronts.

## Key Features

* Handle many-objective problems with an unlimited number of decision variables and optimization objectives.
* Includes standard built-in benchmarks (DTLZ and DPF series) and known MOEAs (NSGA-III, SPEA, etc.).
* Plug in custom benchmarks and algorithms programmatically without modifying the core code.

## Quick Start

```python
from MoeaBench import moeabench                     # Import MoeaBench

exp = moeabench.experiment()                        # Create an instance of an experiment
exp.benchmark = moeabench.benchmark.DTLZ1()         # Select which benchmark to run in the experiment
exp.moea          = moeabench.moea.NSGA_III()       # Select with MOEA to run in the experiment
exp.run                                             # Run the optimization process
moeabench.pareto(exp.result, exp.pof)               # Plot the 3D pareto surface (found and optmimal) 

from myStuff import my_moea, my_bench               # Import your own MOEA and benchmarks
moeabench.add_moea(my_moea)                         # Plug them into moeabench

exp2 = moeabench.experiment()                       # Then proceed as usual  to
exp2.moea      = moeabench.moea.my_moea()           # select your custom MOEA,
exp2.behchmark = moeabench.benchmark.my_benchmark() # select your custom benchmark
exp2.run                                            # and run the optmization experiment.
```

Read more about other MoeaBench capabilties in `docs/manual.md`.

## Contributing

You are more than welcome to contribute.

## License

MoeaBench is free software distributed under the GNU GPL v3 license.

## Contact

For contacting the authors, see file `AUTHORS`.
