# API Renaming Proposal (Alpha Breakage Allowed)

## Direction

- `metrics` / `stats` / `diagnostics`: nomes semânticos da **medida**.
- `view`: nomes pelo **tipo de gráfico** (não pela semântica do fenômeno).
- Remover prefixos de domínio (`perf_`, `topo_`, `strat_`, `clinic_`) quando possível.

---

## Proposed View Grammar

- `view.history(...)`: gráfico temporal (linhas por geração).
- `view.spread(...)`: contraste de distribuições (boxplot / violino).
- `view.density(...)`: densidade (KDE / hist).
- `view.topology(...)`: geometria espacial (scatter/surface).
- `view.bands(...)`: bandas de atingimento / corredores.
- `view.gap(...)`: diferença espacial entre duas superfícies/frentes.
- `view.ranks(...)`: barras de ranks.
- `view.caste(...)`: distribuição hierárquica por rank.
- `view.tiers(...)`: contraste de tiers entre métodos.
- `view.radar(...)`: radar de scores.

---

## Mapping (Current -> Suggested)

### Analysis APIs (`metrics`, `stats`, `diagnostics`)

| Namespace | Nome Atual | Nova Sugestao | Objeto de Entrada | Objeto de Retorno | Views Aplicaveis |
|---|---|---|---|---|---|
| `metrics` | `hv` | `hv` | `Population` | `MetricMatrix` | `view.history`, `view.spread`, `view.density` |
| `metrics` | `gd` | `gd` | `Population` | `MetricMatrix` | `view.history`, `view.spread`, `view.density` |
| `metrics` | `gdplus` | `gdplus` | `Population` | `MetricMatrix` | `view.history`, `view.spread`, `view.density` |
| `metrics` | `igd` | `igd` | `Population` | `MetricMatrix` | `view.history`, `view.spread`, `view.density` |
| `metrics` | `igdplus` | `igdplus` | `Population` | `MetricMatrix` | `view.history`, `view.spread`, `view.density` |
| `metrics` | `emd` | `emd` | `Population` | `MetricMatrix` | `view.history`, `view.spread`, `view.density` |
| `metrics` | `front_size` | `front_ratio` | `Population` | `MetricMatrix` | `view.history`, `view.spread`, `view.density` |
| `stats` | `perf_evidence` | `perf_compare(method='mannwhitney')`<br>Alias: `perf_shift` | `MetricMatrix` | `PerfCompareResult` | `view.spread`, `view.density` |
| `stats` | `perf_distribution` | `perf_compare(method='ks')`<br>Alias: `perf_match` | `MetricMatrix` | `PerfCompareResult` | `view.spread`, `view.density` |
| `stats` | `perf_probability` | `perf_compare(method='a12')`<br>Alias: `perf_win` | `MetricMatrix` | `PerfCompareResult` | `view.spread` |
| `stats` | `topo_distribution` | `topo_compare(method='ks' \| 'emd' \| 'anderson')`<br>Aliases: `topo_match` (KS), `topo_shift` (EMD), `topo_tail` (Anderson) | `Population` | `DistMatchResult` | `view.density` |
| `stats` | `topo_attainment` | `attainment` | `Population` | `AttainmentSurface` | `view.bands`, `view.topology` |
| `stats` | `topo_gap` | `attainment_gap` | `Population` | `AttainmentDiff` | `view.gap` |
| `stats` | `rank_distribution` | `ranks` | `Population` | `RankCompareResult` | `view.ranks` |
| `stats` | `caste_distribution` | `caste` | `Population` | `CasteCompareResult` | `view.caste` |
| `stats` | `tier_duel` | `tiers` | `Population` | `TierResult` | `view.tiers` |
| `clinic` | `audit` | `audit` | `Population` | `DiagnosticResult` | `view.radar`, `view.ecdf`, `view.density`, `view.history` |

### View APIs (`view`)

| Namespace | Nome Atual | Nova Sugestao | Objeto de Entrada |
|---|---|---|---|
| `view` | `topo_shape` | `topology` | `Population` |
| `view` | `topo_bands` | `bands` | `Population` |
| `view` | `topo_gap` | `gap` | `Population` |
| `view` | `topo_density` | `density` | `Population` |
| `view` | `perf_history` | `history` | `MetricMatrix` |
| `view` | `perf_spread` | `spread` | `MetricMatrix` |
| `view` | `perf_density` | `density` | `MetricMatrix` |
| `view` | `strat_ranks` | `ranks` | `RankCompareResult` |
| `view` | `strat_caste` | `caste` | `CasteCompareResult` |
| `view` | `strat_tiers` | `tiers` | `TierResult` |
| `view` | `clinic_ecdf` | `ecdf` | `DiagnosticResult` |
| `view` | `clinic_distribution` | `density` | `DiagnosticResult` |
| `view` | `clinic_history` | `history` | `DiagnosticResult` |
| `view` | `clinic_radar` | `radar` | `DiagnosticResult` |

## Naming Notes (This Revision)

- Metrics clássicas do domínio foram mantidas: `hv`, `gd`, `gdplus`, `igd`, `igdplus`, `emd`.
- `front_size` -> `front_ratio`: mantem consistencia do termo "front" para populacao nao-dominada e explicita proporcao (0..1).
- `perf_compare` vira o guarda-chuva para comparacoes de performance por metodo tecnico (`mannwhitney`, `ks`, `a12`).
- `win_probability` foi simplificado para `win`.
- `topo_compare` vira o guarda-chuva para comparacoes topologicas por metodo tecnico (`ks`, `emd`, `anderson`).
- `diagnostics` publico enxuto: manter apenas `audit`; metricas unitarias ficam fora da API principal.
- `audit` suporta modo completo e parcial: `audit(target, quality=True|False)`.

---

## Stratification Design Decision

A proposta final segue o mesmo pipeline do restante da API:

- `res = mb.stats.ranks(...)`
- `res = mb.stats.caste(...)`
- `res = mb.stats.tiers(...)`
- `res.report()`
- `mb.view.ranks(res)` / `mb.view.caste(res)` / `mb.view.tiers(res)`

### Contract (Recommended)

- Entrada: `Population` (ou compatível).
- Métodos:
  - `mb.stats.ranks(...)`
  - `mb.stats.caste(..., metric=mb.metrics.hv)`
  - `mb.stats.tiers(...)`
- Saídas:
  - `RankCompareResult`
  - `CasteCompareResult`
  - `TierResult`

Prós:
- Mantém o padrão `stats -> report -> view`.
- Permite `caste(..., metric=...)` de forma explícita.
- Garante que o `view` receba um objeto já com o conteúdo visual necessário.

---

## Short Recommendation

- Exponha `mb.stats.ranks`, `mb.stats.caste` e `mb.stats.tiers`.
- Faça as três views consumirem esses resultados diretamente.
- `mb.stats.strata` deixa de ser parte da API canônica pública.
- Em `mb.diagnostics`, exponha publicamente apenas `audit`.
