# Issue #81 — ECDF compression study, 2026-11-12

Noncertified research. See [REPORT.md](REPORT.md) for methodology, tables, five explicit answers and reproduction commands; proposed ADR 0055 records the scientific status.

- `summary.json`: definitions, all parameters/seeds, per-reference hashes, source/environment provenance, full Q matrices, error distributions, ranking summaries, paired variance identities and optimizer seed ranges.
- `q-by-cell.csv.gz`: all 7680 Q evaluations and absolute/relative/signed errors; deterministic gzip.
- `pairwise.json.gz`: all within-K pair gaps for every calibration/method/resolution, full comparator, historical gaps, strict reversals and numerical ties; deterministic gzip.
- `method-comparison.csv`: readable global error/ranking table.
- `reconstructed-full-references.npz`: 8×3367 sorted corrected negative samples per K (`K40`, `K100`, `K200`), with multiplicities retained. Row index matches the original calibration replicate. Original sigma/epsilon/blur seed and per-vector hash are in the summary.
- `execution.log`, `tests.log`, `reproduction.log`: observed local execution evidence.
- `integrity-check.txt`: supplementary environment error/correction and final integrity-check result.
- `manifest.json`: SHA256 and byte sizes of generated data artifacts. The report and execution logs are explanatory evidence outside that generated-data manifest.

Input is the immutable sibling directory `../2026-11-12`, verified against its manifest. No optimizer execution, sigma fitting or fresh ideal sampling is performed. Output m3367 is exactly the full empirical comparator, not an independent condition. No official reference is installed.
