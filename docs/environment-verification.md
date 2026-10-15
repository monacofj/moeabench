# Environment verification

MoeaBench separates environment identity, numerical stability and experimental quality.

- `mb.system.verify_version()` compares the current runtime with the developer-certified dependency manifest. The first `exp.run()` performs the comparison automatically and caches the result; it does not automatically run expensive numerical tests.
- `mb.system.verify_stability()` explicitly tests Python/NumPy PRNG signatures (exact) and stored MOEA smoke fronts (tight numerical tolerances). Passing is evidence for these tested behaviors only, **not** proof that every experiment will replay bit-for-bit.
- `mb.system.verify_experiment(exp)` compares current versions with execution-time provenance attached to an experiment. `exp.load()` performs this comparison automatically when provenance is present. Older archives without provenance are reported as UNKNOWN, not MATCH.
- `exp.report()` summarizes these three states without initiating expensive tests or producing repeated warnings.
- Environment changes invalidate session stability certification. Even after numerical checks pass, experiment provenance differences remain visible.

## What the results establish

Version status (`MATCH`, `MISMATCH`, `UNCONFIGURED`) describes identity
relative to the bundled reference; it does not establish numerical parity.
The explicitly requested stability result (`PASSED` or `FAILED`) covers
only packaged PRNG signatures and representative MOEA fronts.
Experiment provenance (`MATCH`, `MISMATCH`, `UNKNOWN`) independently
describes whether an archived experiment was executed under the current
version fingerprint.

```python
import moeabench as mb

mb.system.verify_version()
mb.system.verify_stability()
mb.system.verify_experiment(exp)  # after execution or loading
```

A different environment may pass numerical checks, and matching dependency
versions can still yield diverging optimization trajectories. In a documented
NSGA-III/DPF1 investigation, differences in energy reference directions of
about `1e-15` were amplified into different evolutionary trajectories.
Rounding the directions in an isolated diagnostic removed the divergence;
the production wrapper does **not** round them. The exact source of the
low-level perturbation remains unresolved. See
[ADR 0050](adr/0050-certified-environment-verification-and-provenance.md).

Matplotlib remains outside the certified numerical dependency manifest;
functional plotting compatibility is tested in the integration suite.

## Developer certification

From the developer reference environment, run `python test.py --deep` (unit, integration and stability test suites). Then run `python test.py --certify`, which reruns the complete deep suite and updates `moeabench/diagnostics/resources/certified_environment.json` **only on success**. Review and commit this manifest with the appropriate release; it does not overwrite numerical baselines.

Certified versions are not inferred from package dependency lower bounds or a random user's installation. If no developer-certified manifest has been committed, the status is UNCONFIGURED and certification is pending.

The canonical front fixtures shipped in the installed wheel mirror the smoke reference fixtures used in the development suite. Developers must review reference changes, not refresh failing fixtures automatically.
