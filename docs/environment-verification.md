# Environment verification

MoeaBench separates environment identity, numerical stability and experimental quality.

- `mb.system.verify_version()` compares the current runtime with the developer-certified dependency manifest. The first `exp.run()` performs the comparison automatically and caches the result; it does not automatically run expensive numerical tests.
- `mb.system.verify_stability()` explicitly tests Python/NumPy PRNG signatures (exact) and stored MOEA smoke fronts (tight numerical tolerances). Passing is evidence for these tested behaviors only, **not** proof that every experiment will replay bit-for-bit.
- `mb.system.verify_experiment(exp)` compares current versions with execution-time provenance attached to an experiment. `exp.load()` performs this comparison automatically when provenance is present. Older archives without provenance are reported as UNKNOWN, not MATCH.
- `exp.report()` summarizes these three states without initiating expensive tests or producing repeated warnings.
- Environment changes invalidate session stability certification. Even after numerical checks pass, experiment provenance differences remain visible.

## Developer certification

From the developer reference environment, run `python test.py --deep` (unit, integration and stability test suites). Then run `python test.py --certify`, which reruns the complete deep suite and updates `moeabench/diagnostics/resources/certified_environment.json` **only on success**. Review and commit this manifest with the appropriate release; it does not overwrite numerical baselines.

Certified versions are not inferred from package dependency lower bounds or a random user's installation. If no developer-certified manifest has been committed, the status is UNCONFIGURED and certification is pending.

The canonical front fixtures shipped in the installed wheel mirror the smoke reference fixtures used in the development suite. Developers must review reference changes, not refresh failing fixtures automatically.
