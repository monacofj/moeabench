# Methods Mapping: Analysis, Reports, and Views

| Namespace | Método(s) | Retorno principal | Tem `.report()`? | Equivalente em `view` |
|---|---|---|---|---|
| `metrics` | `hv`, `gd`, `gdplus`, `igd`, `igdplus`, `emd`, `front_size` | `MetricMatrix` | Sim | `view.perf_history(metric=...)`, `view.perf_spread(metric=...)`, `view.perf_density(metric=...)` |
| `stats` | `perf_evidence`, `perf_distribution` | `HypothesisTestResult` | Sim | `view.perf_spread`, `view.perf_density` |
| `stats` | `topo_distribution` | `DistMatchResult` | Sim | `view.topo_density` |
| `stats` | `strata` | `StratificationResult` | Sim | `view.strat_ranks`, `view.strat_caste`, `view.strat_tiers` |
| `stats` | `tier` | `TierResult` | Sim | `view.strat_tiers` |
| `stats` | `perf_probability`, `emd` | `SimpleStatsValue` | Sim | usado como suporte em views de comparação |
| `stats` | `topo_attainment` | `AttainmentSurface` | Não (é array enriquecido) | `view.topo_bands`, `view.topo_shape` |
| `stats` | `topo_gap` | `AttainmentDiff` | Sim | `view.topo_gap` |
| `diagnostics` | `audit` | `DiagnosticResult` | Sim | `view.clinic_radar` (usa audit), e família `clinic_*` |
| `diagnostics` | `fair_audit` | `FairAuditResult` | Sim | `view.clinic_history/ecdf/distribution` (via métricas FAIR) |
| `diagnostics` | `q_audit` | `QualityAuditResult` | Sim | `view.clinic_radar` |
| `diagnostics` | `headway`, `closeness`, `coverage`, `gap`, `regularity`, `balance` | `FairResult` | Sim | `view.clinic_ecdf`, `view.clinic_distribution`, `view.clinic_history` |
| `diagnostics` | `q_headway`, `q_closeness`, `q_coverage`, `q_gap`, `q_regularity`, `q_balance` | `QResult` | Sim | `view.clinic_radar` (agregado) |
| `diagnostics` | `q_headway_points`, `q_closeness_points` | `np.ndarray` | Não | `view.topo_shape(markers=True)` usa internamente |

## Nota rápida

O princípio "objeto analítico com `.report()` + equivalente visual em `view`" está majoritariamente implementado.  
Exceções naturais: retornos de baixo nível (`np.ndarray`) e renderizadores diretos de figura.
