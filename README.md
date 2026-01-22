<!--
SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# MoeaBench - Multi-objective Evolutionary Algorithm Benchmark

[![REUSE status](https://api.reuse.software/badge/github.com/monacofj/moeabench)](https://api.reuse.software/info/github.com/monacofj/moeabench)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)


> Copyright (c) 2025 Monaco F. J. <monaco@usp.br>  
> Copyright (c) 2025 Silva F. F. <fernandoferreira.silva42@usp.br>  
>
>This project is distributed under the GNU General Public License v3.0 or later. See the file LICENSE for more information. Some third-party components or specific files may be licensed under different terms. Please, consult the SPDX identifiers in each file's header and the LICENSES/ directory for precise details. 



## Introduction

MoeaBench is an **extensible analytical toolkit** that complements multi-objective optimization research by adding a layer of data interpretation and visualization over standard benchmark engines. It achieves this by organizing stochastic search data into a structured semantic model and transforming raw performance metrics into descriptive, narrative-driven results.

The package offers facilities to programmatically create benchmark programs and extract standard performance measurements. Additionally, it provides graphical capabilities to produce time-plots of algorithm convergence and 3D visualizations of Pareto fronts.

## Key Features

*   **Plugin Architecture**: Seamlessly plug in your own algorithms (MOEAs) and problems (MOPs) without modifying the core library. Your custom code is the guest, MoeaBench is the host.
*   **Many-Objective Readiness**: Full support for Many-Objective Optimization (MaOO) with no artificial limits on the number of objectives ($M$) or variables ($N$).
*   **Performance Stability**: Built-in specialized evaluators that switch between exact metrics and efficient approximations (e.g., Monte Carlo) as complexity increases.
*   **Scientific Taxonomy**: Organized analytical perspectives (Topography, Performance, Stratification) for a deep understanding of search dynamics.
*   **Built-in Suite**: Includes standard DTLZ and DPF problem sets and state-of-the-art algorithms via Pymoo integration.
*   **Scientific Rigor**: Native, high-performance implementations of foundational benchmarks (**DTLZ** and **DPF**), rigorously validated against the original literature and audited as the project's analytical "ground truth".

## Quick Start

```python
from MoeaBench import mb                        # Import MoeaBench

exp = mb.experiment()                           # Create an instance of an experiment

exp.mop  = mb.mops.DTLZ2()                      # Select which MOP to run
exp.moea = mb.moeas.NSGA3()                     # Select which MOEA to run

exp.run()                                       # Run the optimization process

mb.view.spaceplot(exp.front())                  # Plot the 3D pareto front
```

## Documentation

*   **[User Guide](docs/userguide.md)**: A comprehensive "How-to" guide covering installation, basic usage, advanced configuration, and custom extensions.
*   **[API Reference](docs/reference.md)**: Exhaustive technical specification of the Data Model, Classes, and Functions.
*   **[API Sheet](docs/api_sheet.py)**: A concise sheet demonstrating syntax and common workflows.

## Examples

MoeaBench provides a collection of documented scripts and Jupyter Notebooks demonstrating various features, from basic runs to advanced population diagnostics:

*   **[Examples Directory](examples/)**: Access all `.py` and `.ipynb` files.
*   Core Examples: `example_01.py` to `example_10.py` cover the essential analytical and diagnostic journey.

## Research & Citation

If you use MoeaBench in your research, please cite the framework using the following metadata:

*   **[Citation File (CITATION.cff)](CITATION.cff)**: Modern machine-readable citation format.


## Contributing

MoeaBench authors warmly welcome community contributions to the project. If you find any bugs or have suggestions for new features, please refer to the **[CONTRIBUTING.md](docs/CONTRIBUTING.md)** file for more information.


## Contact

For contacting the authors, see file `AUTHORS`.

## References

*   **[DTLZ]** K. Deb, L. Thiele, M. Laumanns, and E. Zitzler. "[Scalable multi-objective optimization test problems](https://doi.org/10.1109/CEC.2002.1007032)." *Proc. IEEE Congress on Evolutionary Computation (CEC)*, 2002.
*   **[DPF]** L. Zhen, M. Li, R. Cheng, D. Peng, and X. Yao. "[Multiobjective test problems with degenerate Pareto fronts](https://doi.org/10.48550/arXiv.1806.02706)." *IEEE Transactions on Evolutionary Computation*, vol. 22, no. 5, 2018.
