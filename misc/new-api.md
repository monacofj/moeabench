# Canonical API Grammar

## Direction

- `metrics`, `stats`, and `clinic` are named by **analytical meaning**.
- `view` is named by **chart type**.
- Canonical workflows should prefer:
  - `import moeabench as mb`
  - canonical namespaces under `mb.*`
  - rich result objects followed by `.report()` and correlated `mb.view.*`

---

## Canonical View Grammar

- `mb.view.history(...)`: temporal trajectories
- `mb.view.spread(...)`: comparative distributions
- `mb.view.density(...)`: density / morphology
- `mb.view.topology(...)`: spatial geometry
- `mb.view.bands(...)`: attainment corridors
- `mb.view.gap(...)`: spatial difference between attainment surfaces
- `mb.view.ranks(...)`: rank occupancy
- `mb.view.strata(...)`: rank-wise quality distribution
- `mb.view.tiers(...)`: comparative tier duel
- `mb.view.radar(...)`: diagnostic radar
- `mb.view.ecdf(...)`: diagnostic empirical CDF

---

## Canonical Analysis Surface

| Domain | Analysis API | Canonical Result | Correlated Views |
|---|---|---|---|
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

---

## Semantic Aliases Kept Intentionally

These aliases are part of the intended public language, not compatibility leftovers.

### Metrics

- `mb.metrics.hv` == `mb.metrics.hypervolume`

### Performance Compare

- `mb.stats.perf_shift(...)` == `mb.stats.perf_compare(..., method='mannwhitney')`
- `mb.stats.perf_match(...)` == `mb.stats.perf_compare(..., method='ks')`
- `mb.stats.perf_win(...)` == `mb.stats.perf_compare(..., method='a12')`

### Topography Compare

- `mb.stats.topo_match(...)` == `mb.stats.topo_compare(..., method='ks')`
- `mb.stats.topo_shift(...)` == `mb.stats.topo_compare(..., method='emd')`
- `mb.stats.topo_tail(...)` == `mb.stats.topo_compare(..., method='anderson')`

---

## Stratification Design Decision

The structural ontology is **layer** internally.

The canonical public analytical views over that ontology are:

- `mb.stats.ranks(...)`
- `mb.stats.strata(...)`
- `mb.stats.tiers(...)`

With the corresponding visual endpoints:

- `mb.view.ranks(...)`
- `mb.view.strata(...)`
- `mb.view.tiers(...)`

Recommended workflow:

```python
res = mb.stats.strata(exp, metric=mb.metrics.hv)
res.report()
mb.view.strata(res)
```

This preserves the library-wide pattern:

- compute in `metrics`, `stats`, or `clinic`
- inspect the rich result
- visualize through `mb.view`

---

## Public Diagnostics Policy

The canonical diagnostic entry point is:

- `mb.clinic.audit(target, quality=True)`

Advanced FAIR and Q-score functions may remain public, but they are not the primary entry path for first-time users. They should be treated as advanced public methods rather than the central narrative of the API.

---

## Namespace Policy

- Use `import moeabench as mb`
- Canonical surface lives under:
  - `mb.metrics`
  - `mb.stats`
  - `mb.view`
  - `mb.clinic`
  - `mb.system`
- The canonical experiment constructor is:
  - `mb.experiment(...)`
- `mb.Run` remains public as a model type
- `mb.Experiment` is not part of the public canonical API

