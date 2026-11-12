# Issue #81 — DPF5 complete calibration and paired Q evidence

Noncertified research for proposed ADR 0055 and Draft PR #86. No candidate is installed or released. See ADR 0055 for scientific interpretation and limitations.

- `original-pilot-invalid-independence.json.gz`: original observed pilot; its held-out RNG leakage makes the independence claim invalid.
- `corrected-pilot.json.gz`: same two authentic optimizer populations after reserving all six fitting streams.
- `expanded-real-q.json.gz`: 24 complete calibration draws, 48 authentic runs; includes normalized P0/Pf, distances, all candidate ECDFs, seeds, hashes, environment and 61 protected-file hashes.
- `expanded-analysis.json.gz`: signed changes, per-pair ties/reversals, calibration spread, optimizer spread within algorithm/budget, component sensitivity, full reference rebuilding, clinical reaudits, independent SciPy W1 check and compression sensitivity.
- `real-q-by-run.csv`: readable historical / eight individual / pooled Q values for all 48 runs.
- `manifest.json`: original-byte and compressed SHA256. Gzip timestamps fixed to zero. Producer source at execution is in commit `ccb7445`; the analysis was subsequently extended to investigate compression. All seeds are explicit. Full outputs are retained unchanged.

**Metadata caveat in the expanded original JSON:** legacy top-level `optimizer="NSGA2"` and `generations=300` were inherited from the original single-condition schema. The authoritative design is `algorithms=["NSGA2","SPEA2"]`, `generation_budgets=[40,150]` and the optimizer/generations in each of the 48 rows. That reporting defect is corrected in the current producer; archived bytes are preserved. NSGA-II stores G+1 histories, SPEA2 G; these are generation budgets, not equal evaluation budgets across engines.

Reproduce on the research branch with its installed dependencies:

```bash
MPLCONFIGDIR=/tmp/issue81-mpl .venv/bin/python calibration/scripts/study_dpf5_real_q_full_calibrations_issue81.py \
  --K 40 --K 100 --K 200 --replicates 8 --ideal-size 16384 \
  --budget 40 --budget 150 --seed 1 --seed 7 --seed 19 --seed 43 \
  --algorithm NSGA2 --algorithm SPEA2 --output /tmp/issue81-expanded-reproduction.json
MPLCONFIGDIR=/tmp/issue81-mpl .venv/bin/python calibration/scripts/analyze_dpf5_real_q_issue81.py \
  /tmp/issue81-expanded-reproduction.json --verify --output /tmp/issue81-expanded-reproduction-analysis.json
MPLCONFIGDIR=/tmp/issue81-mpl .venv/bin/python -m pytest \
  tests/unit/test_issue81_full_calibration_q.py \
  tests/unit/test_dpf5_closeness_protocol_issue81.py \
  tests/unit/test_outward_half_normal_blur_issue83.py \
  tests/unit/test_qscore_ecdf.py tests/unit/test_fair_conformance_issue75.py -q
```

Use fresh output paths; producers refuse existing/protected destinations. Numeric results are reproducible in the recorded environment; whole-file hashes can differ with source provenance, environment metadata or the corrected top-level schema. To verify archived evidence without rerunning optimizers, decompress `expanded-real-q.json.gz` to a fresh temporary file and run the analysis command with `--verify`.
