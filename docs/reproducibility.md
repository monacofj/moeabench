<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# Scientific Reproducibility & RNG Stability in MoeaBench

Reproducibility is a pillar of computational research. MoeaBench aims to make reproducibility claims auditable by recording provenance, explicitly checking reference numerical behaviors, and identifying environmental mismatches. The framework does not guarantee exact replay across arbitrary platforms or library updates.

## Environment verification: identity, behavior, and provenance

The strongest reproducibility claim supported by MoeaBench is **conditional
on the behaviors actually tested**. A fixed seed and matching dependency
versions do not by themselves prove bit-for-bit identity on other machines.
In particular, extremely small changes in NSGA-III energy reference
directions have been observed to affect later evolutionary selection.

The environment verification interface distinguishes:

- **Certified version identity:** `mb.system.verify_version()` compares
  Python, MoeaBench, and selected numerical dependency versions with the
  developer-approved manifest. `MATCH` is version agreement, *not* a
  numerical guarantee. `MISMATCH` and `UNCONFIGURED` are diagnostic states.
- **Observed numerical stability:** `mb.system.verify_stability()` runs
  explicit tests of PRNG signatures and selected canonical optimization
  fronts. `PASSED` provides evidence limited to those tests, not all
  algorithms, configurations, or numerical libraries.
- **Experiment-specific provenance:** the environment present when an
  experiment was executed is retained with its archive.
  `mb.system.verify_experiment(exp)` compares this record with the
  current environment. A historical archive without the metadata has
  status `UNKNOWN`, never an inferred match.

`exp.run()` performs the inexpensive version comparison automatically and
advises about unverified environments; it does **not** run stability tests.
`exp.report()` summarizes cached states without triggering expensive work.
See [Environment verification](environment-verification.md) and
[ADR 0050](adr/0050-certified-environment-verification-and-provenance.md).

**Important limitation.** The earlier wording in this document about
universal "exactly reproducible" and "100% blindagem" is an *aspiration*,
not a guarantee for every runtime, architecture or stochastic optimizer.
Use the actual stability and provenance evidence when making scientific
claims. Matplotlib is checked for functional compatibility, not used to
define the numerical certification fingerprint.

## 1. Bit-for-Bit Stability as a Testable Goal

For many multi-objective optimization experiments, a "statistically similar" result is not enough; researchers often require exact bit-for-bit identity between the original run and the validation audit. 

### Why is this difficult?
Standard pseudo-random number generators (PRNGs) evolve. As libraries like NumPy and Python optimize their underlying algorithms (e.g., transitioning from Mersenne Twister to PCG64), the exact sequence of numbers generated from a fixed seed may change.

## 2. Our Implementation Strategy

MoeaBench employs three complementary mechanisms to improve reproducibility and detect drift:

### A. Legacy PRNG for Diagnostics
The clinical diagnostic engine (FAIR Metrics and Q-Scores) uses the **Legacy NumPy RandomState API** (`np.random.RandomState`). 
- **Rationale**: NumPy has explicitly "frozen" this API to ensure bit-of-bit stability for long-term research. Using `RandomState` reduces the risk of PRNG algorithm changes, but does not by itself guarantee identical results across all future numerical environments.

### B. Local RNG state for Algorithms
To prevent interference between parallel experiments or global state side effects, MoeaBench MOEAs (like `NSGA2deap`) utilize **localized generator instances** (e.g., `random.Random(seed)`).
- **Rationale**: Relying on the global `random.seed()` is vulnerable to other libraries or user scripts altering the global state mid-execution. Localized state ensures that each algorithm instance is an isolated, deterministic unit.

### C. Deterministic Component Seeds

Some algorithms contain multiple independent stochastic components. MoeaBench
retains one universal run seed and derives internal component seeds from the run
seed plus a stable, namespaced component identifier. The SHA-256 derivation is
stateless: it does not consume the evolutionary RNG, and adding or reordering
other components does not change an existing component's seed.

NSGA-III uses this policy for energy reference directions:

```python
algorithm = mb.moeas.NSGA3(seed=42)
```

The evolutionary process receives `42`; the reference-direction component
receives a distinct deterministic value recorded on the completed run:

```python
run.seed
# 42

run.component_seeds
# {'nsga3.reference_directions': 286848756}
```

Ordinary repetitions derive new component seeds from each scheduled run seed.
To isolate the evolutionary effect while reusing exactly the same directions,
set the NSGA-III-specific constructor keyword:

```python
algorithm = mb.moeas.NSGA3(seed=42, ref_dirs_seed=7)
```

`ref_dirs_seed=None` restores automatic derivation. A fixed override changes
the reference directions without changing the seed passed to the evolutionary
process; the resulting trajectory may nevertheless change because NSGA-III uses
those directions during selection. Component seeds are included in experiment
reports, persisted archives, and `metadata.json`.

### D. Environment Metadata Capture
Every diagnostic biopsy performed via `mb.clinic.audit()` automatically captures **Reproducibility Metadata**:
- Python version
- NumPy version
- MoeaBench version
- Platform (OS and architecture)
- Generation timestamp

This information is persisted in exported JSON reports, allowing future auditors to reconstruct the exact environment.

## 3. Best Practices for Researchers

To strengthen the evidence for reproducibility in scientific articles, follow these guidelines:

1.  **Always Record Seeds**: Use the `seed` parameter in all MOEA and Experiment initializations.
    For controlled NSGA-III studies, also report the effective value from
    `run.component_seeds` or preserve the experiment archive.
2.  **Save the Environment**: Include a `requirements.txt` or `environment.yml` that specifies the exact version of Python and NumPy used.
3.  **Persist Ground Truths**: While MoeaBench provides analytical PF functions, using fixed "Sidecar" JSON files ensures that any future changes to analytical MOP definitions don't invalidate your historical baselines.
4.  **Export Diagnostic JSONs**: Use the built-in export features to save the full audit trail including the reproducibility metadata block.

## 4. Baseline Version Mapping & Compatibility

To protect against RNG implementation shifts in the underlying language or libraries, MoeaBench implements a **Fail-Safe Baseline Protocol**:

### A. Environment DNA
The canonical `baselines.json` contains "Environment DNA" captured at the moment of calibration:
- `python_version`
- `numpy_version`
- `moeabench_version`

### B. Compatibility Checks
When performing an audit, MoeaBench compares your current session environment against the baseline's DNA:
- **Primary Frozen Baseline**: The canonical alias `baselines.json` is allowed to remain frozen at the last certified calibration release. Its MoeaBench version tag is treated as part of the reference artifact and does not need to match the current codebase version.
- **Historical / External Baselines**: If you explicitly load an older or alternative baseline source, version mismatch remains part of the audit trail and may trigger a `ReproducibilityWarning`.
- **Environment Match**: If the current Python or NumPy version differs from the calibration environment, a `ReproducibilityWarning` is issued.
- **Schema Integrity**: If the JSON structure is incompatible with the current library version, an `UndefinedBaselineError` is raised (Fail-Strict).

### C. Longitudinal Auditing
This protocol allows a researcher in 2028 to run **MoeaBench v5.0** against a **v0.13.2 baseline** to maintain historical parity. The system will allow execution but will explicitly log the version mismatch in the audit trail, ensuring full transparency in the peer-review process.

---
*For more technical details on the FAIR Metrics Framework, see [fair_metrics.md](fair_metrics.md).*
