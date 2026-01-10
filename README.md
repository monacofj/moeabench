<!--
SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# MoeaBench

MoeaBench is a Python framework designed for running benchmarks for Multiobjective Evolutionary Algorithms (MOEA).

The package offers facilities to programmatically create benchmark programs and extract standard performance measurements. Additionally, it provides graphical capabilities to produce time-plots of algorithm convergence and 3D visualizations of Pareto fronts.

## Key Features

* Handle many-objective problems with an unlimited number of decision variables and optimization objectives.
* Includes standard built-in benchmarks (DTLZ and DPF series) and known MOEAs (NSGA-III, SPEA, etc.).
* Plug in custom benchmarks and algorithms programmatically without modifying the core code.

## Quick Start

```python
from MoeaBench import mb                        # Import MoeaBench

exp = mb.experiment()                           # Create an instance of an experiment
exp.benchmark = mb.benchmarks.DTLZ2()           # Select which benchmark to run
exp.moea      = mb.moeas.NSGA3()                # Select which MOEA to run

exp.run()                                       # Run the optimization process

mb.spaceplot(exp.front(), mode='static')        # Plot the 3D pareto front
```

Read more about other MoeaBench capabilties in `docs/userguide.md`.

## Contributing

You are more than welcome to contribute.

## License

MoeaBench is free software distributed under the GNU GPL v3 license.

## Contact

For contacting the authors, see file `AUTHORS`.
