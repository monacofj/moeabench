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
| `stats` | `perf_evidence` | `perf_compare(method='shift')`<br>Alias: `perf_shift` | `MetricMatrix` | `PerfCompareResult` | `view.spread`, `view.density` |
| `stats` | `perf_distribution` | `perf_compare(method='match')`<br>Alias: `perf_match` | `MetricMatrix` | `PerfCompareResult` | `view.spread`, `view.density` |
| `stats` | `perf_probability` | `perf_compare(method='win')`<br>Alias: `perf_win` | `MetricMatrix` | `PerfCompareResult` | `view.spread` |
| `stats` | `topo_distribution` | `topo_compare(method='match' \| 'emd' \| 'anderson')`<br>Alias: `topo_match` (KS) | `Population` | `DistMatchResult` | `view.density` |
| `stats` | `topo_attainment` | `attainment` | `Population` | `AttainmentSurface` | `view.bands`, `view.topology` |
| `stats` | `topo_gap` | `attainment_gap` | `Population` | `AttainmentDiff` | `view.gap` |
| `stats` | `strata` | `strata` | `Population` | `StrataResult` | `view.ranks`, `view.caste`, `view.tiers` |
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
| `view` | `strat_ranks` | `ranks` | `StrataResult` |
| `view` | `strat_caste` | `caste` | `StrataResult` |
| `view` | `strat_tiers` | `tiers` | `StrataResult` |
| `view` | `clinic_ecdf` | `ecdf` | `DiagnosticResult` |
| `view` | `clinic_distribution` | `density` | `DiagnosticResult` |
| `view` | `clinic_history` | `history` | `DiagnosticResult` |
| `view` | `clinic_radar` | `radar` | `DiagnosticResult` |

## Naming Notes (This Revision)

- Metrics clássicas do domínio foram mantidas: `hv`, `gd`, `gdplus`, `igd`, `igdplus`, `emd`.
- `front_size` -> `front_ratio`: mantem consistencia do termo "front" para populacao nao-dominada e explicita proporcao (0..1).
- `perf_compare` vira o guarda-chuva para comparacoes de performance por metodo (`shift`, `match`, `win`).
- `win_probability` foi simplificado para `win`.
- `topo_compare` vira o guarda-chuva para comparacoes topologicas por metodo (`match`, `emd`, `anderson`).
- `diagnostics` publico enxuto: manter apenas `audit`; metricas unitarias ficam fora da API principal.
- `audit` suporta modo completo e parcial: `audit(target, quality=True|False)`.

---

## Strata Design Decision

A proposta aqui fica objetiva: **1 método** (`mb.stats.strata`) e **1 tipo de objeto de retorno** (ex.: `StrataResult`), com **3 visualizações** sobre o mesmo objeto.

### Contract (Recommended)

- Entrada: `Population` (ou compatível).
- Método: `mb.stats.strata(...)`.
- Saída: objeto único (`StrataResult`).
- Visualizações suportadas pelo mesmo resultado:
  - `view.ranks(strata_result)`
  - `view.caste(strata_result)`
  - `view.tiers(strata_result)`

Prós:
- Contrato único e consistente para estratificação.
- Evita duplicar cálculos entre visualizações.
- Tabela/API ficam coerentes: uma medida, três lentes visuais.

### Option B (descartada)

- Unificar em `view.caste(..., mode='ranks|strata|tiers')`.

Contras:
- `mode` vira switch grande e mais frágil.
- Perde semântica direta por visualização.

---

## Short Recommendation

- Faça a quebra para `mb.stats.strata(...)` com retorno único (`StrataResult`).
- Faça as três views consumirem esse mesmo objeto.
- Não introduza alias curto `mb.strata()`.
- Em `mb.diagnostics`, exponha publicamente apenas `audit`.
