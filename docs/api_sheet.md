<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
SPDX-FileCopyrightText: 2026 Silva F. F. <fernandoferreira.silva42@usp.br>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# MoeaBench API Map

This document presents the MoeaBench API as an organized **thematic map**.
It is not a replacement for the exhaustive contract in **[reference.md](reference.md)**. Instead, its purpose is to clarify the *structure* of the public API: what belongs to the core, what belongs to analytical domains, how views correlate with results, and which methods are canonical versus advanced.

The canonical usage style is:

```python
import moeabench as mb
```

---

## **Summary**
1.  **[1. Core API](#1-core-api)**
2.  **[2. Data Hierarchy and Access](#2-data-hierarchy-and-access)**
3.  **[3. Analysis Domains](#3-analysis-domains)**
4.  **[4. View Grammar](#4-view-grammar)**
5.  **[5. Reportable Results](#5-reportable-results)**
6.  **[6. Advanced Public Methods](#6-advanced-public-methods)**
7.  **[7. Namespace Roles](#7-namespace-roles)**

---

## **1. Core API**

The **Core API** is the operational backbone of the library. It covers experiment construction, execution, calibration, persistence, export, environment inspection, and global configuration.

| Namespace | Method / Object | Role | Input | Output / Effect |
| :--- | :--- | :--- | :--- | :--- |
| `mb` | `experiment(...)` | Canonical experiment constructor | Problem, algorithm, metadata, kwargs | `Experiment` object |
| `mb` | `Run` | Public trajectory type | Returned by `exp[i]` | `Run` object |
| `mb` | `defaults` | Global configuration registry | Attribute assignment | Global execution / stats / plotting defaults |
| `exp` | `run(...)` | Executes the protocol | `repeat`, `silent`, `stop`, MOEA kwargs | Populates runs/history |
| `exp` | `save(path, mode=...)` | Persists experiments | Path, mode | ZIP archive |
| `exp` | `load(path, mode=...)` | Restores experiments | Path, mode | Rehydrates experiment state |
| `mop` | `calibrate(...)` | Problem-centered clinical calibration | GT/baseline/search sources | Sidecar / baseline artifacts |
| `mb.clinic` | `calibrate(...)` | Procedural calibration entry point | MOP and calibration sources | Same calibration engine as `mop.calibrate(...)` |
| `mb.system` | `version(show=True)` | Version inspection | `show` flag | Version string |
| `mb.system` | `check_dependencies()` | Environment dependency audit | None | Dependency report |
| `mb.system` | `info(show=True)` | Reproducibility metadata | `show` flag | `dict` |
| `mb.system` | `output(text, markdown=None)` | Environment-aware text emission | Text / optional Markdown | Emitted text string |
| `mb.system` | `export_objectives(...)` | CSV export of objective values | Experiment, Run, Population, array | CSV file |
| `mb.system` | `export_variables(...)` | CSV export of decision variables | Experiment, Run, Population, array | CSV file |

> [!NOTE]
> The canonical experiment constructor is `mb.experiment(...)`. `mb.Experiment` is not part of the canonical public API.

---

## **2. Data Hierarchy and Access**

MoeaBench is organized around a data hierarchy. This hierarchy is not just an implementation detail: it explains the polymorphism of almost every analytical call in the library.

| Level | Object | Access Pattern | Semantic Role | Typical Return |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `Experiment` | `exp` | Global manager of all runs | Multi-run container |
| 2 | `Run` | `exp[i]`, `exp.last_run` | One stochastic trajectory | Run-level history |
| 3 | `Population` | `exp.pop(n)`, `run.pop(n)` | Snapshot at one generation | Population of solutions |
| 4 | Objective / Variable Space | `pop.objs`, `pop.vars` | Raw numerical space | `SmartArray` / `np.ndarray`-like |
| 1 -> 3 | Cloud front | `exp.front()` | Non-dominated aggregate across runs | Objective-space front |
| 1 -> 3 | Cloud set | `exp.set()` | Non-dominated variables across runs | Decision-space set |
| 1 -> 3 | Cloud filter | `exp.non_dominated()`, `exp.dominated()` | Aggregate dominance filters | `Population` |
| 2 -> 3 | Local filter | `run.non_dominated()`, `run.dominated()` | Run-local dominance filters | `Population` |
| 1 -> 3 | Analytical truth | `exp.optimal()`, `exp.optimal_front()`, `exp.optimal_set()` | Ground-truth reference | `Population` / `SmartArray` |

This hierarchy underlies the library-wide "Smart Argument" behavior:

```python
import moeabench as mb

exp = mb.experiment()
run = exp[0]
pop = run.pop(-1)

# The same family of APIs can often consume any of these
mm = mb.metrics.hv(exp)
res = mb.stats.strata(pop)
diag = mb.clinic.audit(run)
```

---

## **3. Analysis Domains**

This is the central analytical table of the API. It correlates **analysis namespaces** (`metrics`, `stats`, `clinic`) with their canonical result objects and the views that consume them.

| Domain | Analysis API | Canonical Result | Correlated Views |
| :--- | :--- | :--- | :--- |
| Performance | `mb.metrics.hypervolume` / `mb.metrics.hv` | `MetricMatrix` | `view.history`, `view.spread`, `view.density` |
| Performance | `mb.metrics.gd`, `mb.metrics.gdplus`, `mb.metrics.igd`, `mb.metrics.igdplus`, `mb.metrics.emd`, `mb.metrics.front_ratio` | `MetricMatrix` | `view.history`, `view.spread`, `view.density` |
| Performance Compare | `mb.stats.perf_compare(...)` | `PerfCompareResult` | `view.spread`, `view.density` |
| Performance Compare | `mb.stats.perf_shift`, `mb.stats.perf_match`, `mb.stats.perf_win` | `PerfCompareResult` | `view.spread`, `view.density` |
| Topography Compare | `mb.stats.topo_compare(...)` | `DistMatchResult` | `view.density` |
| Topography Compare | `mb.stats.topo_match`, `mb.stats.topo_shift`, `mb.stats.topo_tail` | `DistMatchResult` | `view.density` |
| Attainment | `mb.stats.attainment(...)` | `AttainmentSurface` | `view.bands`, `view.topology` |
| Attainment Gap | `mb.stats.attainment_gap(...)` | `AttainmentDiff` | `view.gap` |
| Stratification | `mb.stats.ranks(...)` | `RankCompareResult` | `view.ranks` |
| Stratification | `mb.stats.strata(...)` | `StrataCompareResult` | `view.strata` |
| Stratification | `mb.stats.tiers(...)` | `TierResult` | `view.tiers` |
| Diagnostics | `mb.clinic.audit(...)` | `DiagnosticResult` | `view.radar`, `view.ecdf`, `view.density`, `view.history` |

This table defines the canonical workflow pattern:

```python
import moeabench as mb

res = mb.stats.strata(exp, metric=mb.metrics.hv)
res.report()
mb.view.strata(res)
```

The same pattern applies across domains:

- compute in `metrics`, `stats`, or `clinic`
- inspect the rich result
- visualize through `mb.view`

---

## **4. View Grammar**

The `view` namespace is organized by **chart type**, not by the semantics of the analytical phenomenon.

| View | Visual Form | Typical Inputs | Produces | Domain Notes |
| :--- | :--- | :--- | :--- | :--- |
| `mb.view.history(...)` | Line trajectories over generations | `MetricMatrix`, `Experiment`, `Run`, `DiagnosticResult` | Temporal figure | Performance or clinic, depending on input/domain |
| `mb.view.spread(...)` | Boxplot / comparative spread | `PerfCompareResult`, `MetricMatrix`, experiment-like inputs | Comparative distribution figure | Performance-oriented |
| `mb.view.density(...)` | KDE / histogram / density morphology | Performance, topology, or clinic inputs/results | Density figure | Domain-dispatched |
| `mb.view.topology(...)` | Scatter / spatial geometry | `Population`, `Experiment`, `Run`, `AttainmentSurface` | Spatial figure | Objective-space geometry |
| `mb.view.bands(...)` | Attainment corridor | `AttainmentSurface` or experiment-like inputs | Corridor / EAF band figure | Reliability view |
| `mb.view.gap(...)` | Surface difference | `AttainmentDiff` or two experiment-like inputs | Gap figure | Comparative coverage |
| `mb.view.ranks(...)` | Rank occupancy bars | `RankCompareResult` or compatible structure | Rank frequency figure | Layer occupancy |
| `mb.view.strata(...)` | Rank-wise quality distribution | `StrataCompareResult` | Hierarchical boxplot figure | Layer-quality relationship |
| `mb.view.tiers(...)` | Comparative tier duel | `TierResult` | Duel / proportion figure | Joint competition |
| `mb.view.radar(...)` | Spider / radar chart | `DiagnosticResult`, `audit(...)`-compatible input | Diagnostic fingerprint | Holistic clinical summary |
| `mb.view.ecdf(...)` | Empirical cumulative curve | Clinical input / diagnostic result | Goal-attainment figure | Clinical deep inspection |

Two principles govern the `view` namespace:

1.  **Views visualize**. They do not define independent scientific semantics.
2.  **Views accept rich results directly** whenever a canonical result object already exists.

---

## **5. Reportable Results**

MoeaBench is not organized around raw arrays alone. Its public API is strongly centered on **rich result objects** that implement `.report()`.

| Result Type | Produced By | Meaning | Supports `.report()` | Typical View |
| :--- | :--- | :--- | :--- | :--- |
| `MetricMatrix` | `mb.metrics.*` | Metric trajectories over runs and generations | Yes | `history`, `spread`, `density` |
| `PerfCompareResult` | `mb.stats.perf_compare(...)` and semantic aliases | Statistical comparison of performance metrics | Yes | `spread`, `density` |
| `DistMatchResult` | `mb.stats.topo_compare(...)` and semantic aliases | Topological distribution comparison | Yes | `density` |
| `AttainmentSurface` | `mb.stats.attainment(...)` | EAF-style attainment surface | Limited structural reporting | `bands`, `topology` |
| `AttainmentDiff` | `mb.stats.attainment_gap(...)` | Gap between two attainment surfaces | Yes | `gap` |
| `RankCompareResult` | `mb.stats.ranks(...)` | Rank occupancy structure | Yes | `ranks` |
| `StrataCompareResult` | `mb.stats.strata(...)` | Rank-wise quality distribution | Yes | `strata` |
| `TierResult` | `mb.stats.tiers(...)` | Shared-rank duel between methods | Yes | `tiers` |
| `DiagnosticResult` | `mb.clinic.audit(...)` | Clinical synthesis over FAIR and Q-score layers | Yes | `radar`, `ecdf`, `density`, `history` |

This result-oriented design is one of the main architectural signatures of MoeaBench.

---

## **6. Advanced Public Methods**

Some methods are public but are not intended to be the primary entry path for first-time users. They remain important because they expose deeper analytical layers and support advanced workflows, custom plots, and pathology research.

| Namespace | Method | Category | Return | Why Advanced |
| :--- | :--- | :--- | :--- | :--- |
| `mb.clinic` | `headway`, `closeness`, `coverage`, `gap`, `regularity`, `balance` | FAIR / physical diagnostics | `FairResult` | Lower-level physical facts below the clinical synthesis |
| `mb.clinic` | `q_headway`, `q_closeness`, `q_coverage`, `q_gap`, `q_regularity`, `q_balance` | Clinical normalized scores | `QResult` | Direct access to individual clinical layers |
| `mb.clinic` | `q_headway_points`, `q_closeness_points` | Point-wise helpers | `np.ndarray` | Per-point diagnostics and semantic overlays |
| `mb.clinic` | `register_baselines(source)` | Baseline management | Side effect / registry update | Session-level baseline control |
| `mb.clinic` | `reset_baselines()` | Baseline management | `None` | Restores built-in defaults |
| `mb.clinic` | `use_baselines(source)` | Baseline management | Context manager | Temporary baseline switching |

The canonical diagnostic entry point remains:

```python
import moeabench as mb

diag = mb.clinic.audit(exp, quality=True)
diag.report()
mb.view.radar(diag)
```

The advanced methods above should be understood as **bonus depth**, not as the default API narrative for new users.

---

## **7. Namespace Roles**

The public API is intentionally segmented by namespace role.

| Namespace | Responsibility | Canonical or Support | Examples |
| :--- | :--- | :--- | :--- |
| `mb` | Top-level facade | Canonical | `mb.experiment`, `mb.metrics`, `mb.stats`, `mb.view` |
| `mb.metrics` | Scalar metric computation | Canonical | `hv`, `igd`, `emd`, `front_ratio` |
| `mb.stats` | Statistical comparison and structural analysis | Canonical | `perf_compare`, `attainment`, `strata`, `tiers` |
| `mb.view` | Visualization grammar | Canonical | `history`, `density`, `topology`, `radar` |
| `mb.clinic` | Diagnostic synthesis and advanced pathology tools | Canonical entry + advanced public methods | `audit`, `closeness`, `q_closeness`, `use_baselines` |
| `mb.system` | Runtime environment, export, and reproducibility helpers | Canonical support | `info`, `version`, `export_objectives` |
| `mb.mops` | Built-in benchmark problems | Support / domain content | `DTLZ2`, `DPF1`, custom MOP base classes |
| `mb.moeas` | Built-in solver implementations | Support / domain content | `NSGA3`, `MOEAD`, custom MOEA base classes |
| `mb.defaults` | Global configuration | Canonical support | `population`, `generations`, `alpha`, `backend` |

In architectural terms:

- `metrics`, `stats`, and `clinic` name **what is being analyzed**
- `view` names **how it is visualized**
- `system` names **operational support**
- `mops` and `moeas` provide the scientific content that the rest of the API operates on

---

## **See Also**

- **[userguide.md](userguide.md)**: tutorial and didactic walkthrough
- **[reference.md](reference.md)**: exhaustive contracts, arguments, and returns
- **[mops.md](mops.md)**: benchmark families and mathematical background
- **[fair.md](fair.md)**: FAIR metrics and clinical diagnostics internals

