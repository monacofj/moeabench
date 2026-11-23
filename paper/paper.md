---
title: "MoeaBench: A Framework for Reproducible Experimental Analysis of Multi-Objective Evolutionary Algorithms"
tags:
  - Python
  - multi-objective optimization
  - evolutionary algorithms
  - benchmarking
  - reproducible research
authors:
  - name: Fernando Ferreira da Silva
    affiliation: 1
  - name: Alexandre Cláudio Botazzo Delbem
    affiliation: 1
  - name: Abraham Kuruvilla
    affiliation: 2
  - name: Francisco José Monaco
    affiliation: 1
    corresponding: true
affiliations:
  - name: Institute of Mathematical and Computer Sciences, University of São Paulo, Brazil
    index: 1
  - name: Ribeirão Preto Medical School, University of São Paulo, Brazil
    index: 2
date: 11 November 2026
bibliography: paper.bib
---

# Summary

Comparative studies of multi-objective evolutionary algorithms (MOEAs) rely on stochastic experiments whose conclusions depend on randomization, normalization, reference points, and computational environments. MoeaBench is an extensible Python framework for reproducible experimental analysis of MOEAs. It introduces an experiment-centric abstraction that integrates execution, data provenance, quantitative assessment, and interpretation within a common scientific workflow.

The framework organizes experiments into independent runs and successive populations, preserving evolutionary histories and data in both decision and objective spaces. It systematically manages random seeds, normalization scales, reference points, and calibrated baselines, while recording experimental configurations and software environments. Environment verification and numerical stability checks help identify discrepancies associated with changes in runtime conditions or numerical implementations.

Alongside classical performance indicators, MoeaBench provides original analytical methods for studying population structure and diagnosing Pareto approximations, complemented by empirical attainment analysis and rigorous statistical comparisons. Structured analytical results connect numerical measurements to visualizations and explanatory reports, supporting interpretable examination of algorithm behavior. Implementations of benchmark problems and explicit plugin contracts allow researchers to incorporate their own problems and algorithms without modifying the framework.

# Statement of need

Research on multi-objective evolutionary algorithms (MOEAs) requires controlled experiments across benchmark problems, algorithm configurations, and repeated independent runs. Reliable evaluation depends on methodological choices extending beyond the algorithms themselves, including random seed management, reference fronts and points, normalization scales, performance indicators, statistical procedures, and computing environments. These choices can substantially affect experimental conclusions. For example, hypervolume values computed using different reference points may lead to inconsistent comparisons, while changes in numerical libraries can alter optimization trajectories even under identical algorithm configurations.

Experimental evidence also extends beyond aggregate performance indicators. Differences in convergence, coverage, solution distribution, and population structure may be obscured by scalar summaries, particularly in many-objective optimization, where high-dimensional objective spaces increase computational costs and complicate performance assessment. Researchers therefore need methods for examining evolutionary histories and solution distributions in both decision and objective spaces, supported by rigorous statistical comparisons across independent runs.

Producing such analyses typically requires coordinating optimization engines, benchmark implementations, data collection, metric computation, statistical analysis, and visualization. When implemented through separate, experiment-specific scripts, these procedures introduce opportunities for programming errors, inconsistent conventions, and undocumented assumptions, hindering reproducibility and independent reanalysis. Addressing these challenges requires a common experimental abstraction that preserves provenance, makes methodological choices explicit, and supports consistent analytical procedures while accommodating original optimization problems and algorithms. Such an approach provides a foundation for reproducible, comparable, auditable, and interpretable experimental research.

# State of the field

The field of multi-objective evolutionary optimization has a mature software ecosystem supporting algorithm implementations, benchmark problems, and performance assessment [@blank2020pymoo; @benitez2019jmetalpy; @tian2017platemo]. MoeaBench contributes to this ecosystem through an experiment-centric analytical abstraction that integrates experimental execution, data provenance, quantitative evaluation, statistical analysis, and interpretation. By organizing independent runs, population histories, evaluation references, and methodological metadata within a common experimental model, the framework enables systematic investigation of evolutionary behavior across generations and in both decision and objective spaces.

MoeaBench also extends the available analytical repertoire by combining classical performance indicators with original methods for population structure analysis and diagnostics of Pareto approximations. Its capabilities include ordinal hypervolume, statistical performance comparisons using hypothesis tests and effect sizes, distributional and topological comparisons, population stratification by dominance ranks, comparative tiers, and empirical attainment analysis. These complementary methods allow researchers to investigate convergence, coverage, solution distribution, structural differences, and stochastic variability from multiple perspectives.

The integration of these analytical methods within a common experimental model enables their systematic application, comparison, and interpretation. Structured analytical results support explanatory reports and complementary visualizations, while explicit extension contracts allow original optimization problems and algorithms to participate in the same experimental and analytical procedures.

# Software design

MoeaBench is implemented in Python around an object-oriented experimental model. The public `mb.experiment` abstraction encapsulates a multi-objective optimization problem (MOP), a multi-objective evolutionary algorithm (MOEA), execution configuration, and collected results. Each experiment contains independent `Run` objects representing stochastic trajectories, with successive `Population` snapshots preserving the correspondence between decision variables and objective values. Numerical data are represented through NumPy-compatible structures, including `SmartArray`, which carries contextual information such as labels and data origin. This hierarchy supports input-polymorphic analytical methods that operate on experiments, runs, or individual populations, automatically selecting the relevant data while preserving their experimental context.

The execution architecture separates experimental orchestration from optimization engines through adapters, including interfaces to pymoo and DEAP. User-defined problems and algorithms implement explicit `BaseMop` and `BaseMoea` contracts, allowing new components to participate in the same experimental and analytical procedures without modifying the framework. MoeaBench also provides vectorized implementations of classical DTLZ and DPF benchmark problems. Reference Pareto fronts are approximated through problem-specific analytical sampling or numerical methods, with calibration procedures supporting reference data generation and validation. Finite reference sets are not assumed to exhaust the continuous Pareto front.

The analytical architecture separates performance measurement (`metrics`), statistical and structural analysis (`stats`), and visualization (`view`). These components support performance indicators, statistical comparisons, population stratification, and distributional analysis in both decision and objective spaces. Analytical methods return structured results containing numerical measurements, supporting data, and contextual metadata. Their reporting interfaces provide explanatory summaries, while visualization functions can consume the same results to produce complementary spatial, temporal, and statistical perspectives. This separation allows analytical evidence to be reused independently of its presentation. Computational efficiency is addressed through NumPy vectorization, memory-aware processing, and specialized numerical backends. For computationally demanding indicators in many-objective optimization problems (MaOPs), such as hypervolume, the framework selects between exact algorithms and reproducible Monte Carlo approximations according to dimensionality. Absolute hypervolume uses a certified continuous Pareto-front denominator when the problem's geometry and normalization are mathematically verified; otherwise, it retains the finite calibrated reference, whose discretization can yield values above one. The result records which denominator was used. Relative hypervolume measures performance against a contextual competitive reference and can also legitimately exceed one.

Reproducibility is incorporated into the experimental architecture through deterministic seed management, selective persistence, and environment provenance. Independent random streams for internal stochastic components are derived from run seeds using stable component identifiers. Portable experiment archives preserve configurations, results, population histories, and scientific metadata for subsequent analysis. Environment verification distinguishes software-version compatibility, observed numerical stability, and experiment provenance, allowing researchers to identify potential discrepancies without assuming that identical dependency versions guarantee identical numerical behavior. Changes to sensitive dependencies such as pymoo can also be checked through targeted, explicitly invoked behavioral regression tests.

Software validation follows a three-scope testing architecture. Unit tests verify API contracts, mathematical invariants, and local component behavior. Integration tests exercise complete workflows involving execution, analysis, visualization, and persistence. Stability tests compare numerical measurements and optimization results against preserved reference data, detecting regressions in both analytical methods and evolutionary trajectories. A central test runner organizes the tests by scope and applicable validation level (*smoke* and *basic* for numerical stability); independent performance certification evaluates stochastic behavior against a frozen historical reference. A separate, manually invoked battery verifies selected scientifically sensitive external dependency behaviors.

# Research impact statement

Development of MoeaBench began in 2024 as part of research on the experimental evaluation of multi-objective evolutionary algorithms. The framework became the central software artifact of Fernando Ferreira da Silva's 2026 master's dissertation at the University of São Paulo [@silva2026moeabench]. The dissertation documents its architecture and demonstrates its application to comparative studies involving benchmark problems, repeated stochastic runs, performance indicators, and structural and statistical analysis. One study compares NSGA-II and MOEA/D using empirical attainment surfaces across ten independent runs per algorithm to investigate the spatial variability of their results.

MoeaBench is currently used extensively in research on strategies for selecting directions to reach Pareto-front corners and in the development of a subspace-search-based multi-objective evolutionary algorithm, building on earlier research on many-table evolutionary optimization [@brasil2013meamt]. Researchers from other local research groups also use the framework in their optimization studies, extending its application beyond the original development team.

The public repository provides executable examples, benchmark implementations, automated tests, and architectural documentation supporting reproducibility and reuse. External community participation includes a merged code contribution correcting mutation-rate behavior, issue reports, and proposals to improve example validation and extend the available benchmark suites.

# AI usage disclosure

AI assistance tools (OpenAI Codex/GPT-5-Codex, Google Antigravity/Gemini 3, and ChatGPT/GPT-6) were used under the authors' supervision for repository maintenance, code refactoring, defect detection, code auditing, test automation, documentation refinement, and manuscript preparation. The human authors remain fully responsible for the intellectual work, scientific and architectural decisions, and overall software development. AI-assisted outputs were reviewed and edited by the authors, with code manually inspected and rigorously tested using the automated test suite, and manuscript content checked against project documentation and scientific references.

# Acknowledgements

The authors thank Luis Aires for his contributions to the MoeaBench codebase, particularly improvements to mutation-rate handling, and the open-source community members who contributed suggestions, issue reports, and feedback throughout the project's development.

# References
