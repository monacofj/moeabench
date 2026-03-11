# ADR 0038: Runtime Output Control and Sidecar Hygiene (v0.13.3)

**Status:** Accepted  
**Date:** 2026-03-11  
**Author:** Monaco F. J.  
**Drivers:** API Ergonomics, Notebook/CLI Usability, Reproducibility Hygiene.

---

## 1. Context

Recent UX adjustments introduced three coupled concerns:
1. Runtime observability should be explicit (`run()` should announce what is executing).
2. Batch and CI workflows require zero-noise execution.
3. Calibration sidecars are local artifacts and should not pollute version control.

In addition, experiment naming by variable identifier should apply to the `name` property itself, not only report rendering.

---

## 2. Decision

### 2.1 `experiment.run(..., silent=False)`
- `run()` now prints `Running {exp.name}` by default.
- New parameter `silent=True` suppresses run output completely:
  - startup banner,
  - progress bars,
  - run-triggered diagnostic prints.

### 2.2 `mb.system.version(show=True)`
- `show=True` remains the default interactive behavior (prints and returns version).
- `show=False` is the silent retrieval mode.
- Internal framework calls must use `show=False` to avoid accidental output during import/metadata flows.

### 2.3 Name inference promotion
- Variable-name inference is promoted to the `experiment.name` property.
- Unnamed experiments now resolve to their caller identifier (e.g., `exp1`), while explicit user names keep precedence.

### 2.4 Sidecar repository policy
- Calibration sidecars follow `<ProblemName>_M<M>.json`.
- These files are considered local execution artifacts and are ignored by Git (`*_M[0-9]*.json`).

---

## 3. Consequences

- CLI/notebook execution is easier to track by default.
- CI and large sweeps can run silently without output contamination.
- Documentation and examples remain consistent: direct `mb.system.version()` calls and no redundant `print("Running...")`.
- Repositories remain clean from machine-local calibration payloads.

