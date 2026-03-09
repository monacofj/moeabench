# Scientific Reproducibility & RNG Stability in MoeaBench

Reproducibility is a pillar of computational research. MoeaBench is designed with a "Fail-Strict" philosophy to ensure that results published in peer-reviewed literature remain exactly reproducible across time, platforms, and library updates.

## 1. The Bit-for-Bit Stability Standard

For many multi-objective optimization experiments, a "statistically similar" result is not enough; researchers often require exact bit-for-bit identity between the original run and the validation audit. 

### Why is this difficult?
Standard pseudo-random number generators (PRNGs) evolve. As libraries like NumPy and Python optimize their underlying algorithms (e.g., transitioning from Mersenne Twister to PCG64), the exact sequence of numbers generated from a fixed seed may change.

## 2. Our Implementation Strategy

MoeaBench employs three layers of "blindagem" (protection) to lock in reproducibility:

### A. Legacy PRNG for Diagnostics
The clinical diagnostic engine (FAIR Metrics and Q-Scores) uses the **Legacy NumPy RandomState API** (`np.random.RandomState`). 
- **Rationale**: NumPy has explicitly "frozen" this API to ensure bit-of-bit stability for long-term research. By standardizing on `RandomState`, MoeaBench ensures that the numerical targets for a specific version remain identical in all future NumPy releases.

### B. Local RNG state for Algorithms
To prevent interference between parallel experiments or global state side effects, MoeaBench MOEAs (like `NSGA2deap`) utilize **localized generator instances** (e.g., `random.Random(seed)`).
- **Rationale**: Relying on the global `random.seed()` is vulnerable to other libraries or user scripts altering the global state mid-execution. Localized state ensures that each algorithm instance is an isolated, deterministic unit.

### C. Environment Metadata Capture
Every diagnostic biopsy performed via `mb.diagnostics.audit()` automatically captures **Reproducibility Metadata**:
- Python version
- NumPy version
- MoeaBench version
- Platform (OS and architecture)
- Generation timestamp

This information is persisted in exported JSON reports, allowing future auditors to reconstruct the exact environment.

## 3. Best Practices for Researchers

To achieve 100% "Blindagem" in your articles, follow these guidelines:

1.  **Always Record Seeds**: Use the `seed` parameter in all MOEA and Experiment initializations.
2.  **Save the Environment**: Include a `requirements.txt` or `environment.yml` that specifies the exact version of Python and NumPy used.
3.  **Persist Ground Truths**: While MoeaBench provides analytical PF functions, using fixed "Sidecar" JSON files ensures that any future changes to analytical MOP definitions don't invalidate your historical baselines.
4.  **Export Diagnostic JSONs**: Use the built-in export features to save the full audit trail including the reproducibility metadata block.

## 4. Baseline Version Mapping & Compatibility

To protect against RNG implementation shifts in the underlying language or libraries, MoeaBench implements a **Fail-Safe Baseline Protocol**:

### A. Environment DNA
Every baseline JSON file (e.g., `baselines_v0.13.2.json`) contains "Environment DNA" captured at the moment of calibration:
- `python_version`
- `numpy_version`
- `moeabench_version`

### B. Compatibility Checks
When performing an audit, MoeaBench compares your current session environment against the baseline's DNA:
- **Major/Minor Match**: If the current Python or NumPy version differs from the calibration environment, a `ReproducibilityWarning` is issued. 
- **Schema Integrity**: If the JSON structure is incompatible with the current library version, an `UndefinedBaselineError` is raised (Fail-Strict).

### C. Longitudinal Auditing
This protocol allows a researcher in 2028 to run **MoeaBench v5.0** against a **v0.13.2 baseline** to maintain historical parity. The system will allow execution but will explicitly log the version mismatch in the audit trail, ensuring full transparency in the peer-review process.

---
*For more technical details on the FAIR Metrics Framework, see [fair.md](fair.md).*
