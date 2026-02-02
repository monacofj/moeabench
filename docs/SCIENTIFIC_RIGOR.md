# Scientific Rigor & Code Integrity Protocol

This document establishes the binding rules for development and experimentation within the MoeaBench project. These rules are designed to prevent "blind" execution, eliminate unverified assumptions, and ensure that every byte generated is scientifically sound and reproducible.

## 1. The Anti-Presumption Rule
**"Verification over Assumption."**
- Never assume the default behavior of a library, method, or constructor (especially regarding `seeds`, `tolerances`, or `global states`).
- Every critical parameter (random seeds, population sizes, mutation rates) must be explicitly managed or verified via a "Pre-flight Unit Test."
- If a parameter's factual truth is unknown, the agent MUST stop and ask the USER for clarification or conduct an isolated code inspection.

## 2. Invariant-Driven Development (Panic Checks)
**"Fail Fast, Fail Loudly."**
- Every long-running script (Generation, Analysis, Calibration) MUST implement real-time **Panic Checks**.
- **Stochastic Panic**: If subsequent runs of a supposedly random process produce identical results (zero variance), the process must abort immediately with a `RuntimeError`.
- **Mathematical Panic**: If a MOP invariant (e.g., $f_i \le 1.0$ for DTLZ) is violated, the process must abort.
- **Data Integrity Panic**: If `NaN` or `Inf` values are detected in objectives or decision variables, the process must abort.

## 3. The Pre-Flight Audit
**"Double-Read, Single-Run."**
- Before initiating any large-scale task (Phase 1.8, Phase 2, etc.), the agent must perform a deep audit of the proposed code:
    1. **Logic Integrity**: Check for loop boundary errors, variable name shadowing, and hidden state resets.
    2. **Fault Tolerance**: Ensure atomic saving and resumability are correctly implemented and tested for race conditions.
    3. **Performance**: Verify that expensive calculations (like Hypervolume or IGD) are not redundant.
- The results of this audit must be presented as a "Pre-flight Checklist" to the USER.

## 4. Scalability Gatekeeping
**"Sample, Audit, Scale."**
- No large-scale run (N > 10 configs) is permitted without a prior **Small-Scale Audit (Micro-Audit)**.
- A Micro-Audit consists of a minimal valid set (e.g., 2 MOPs, 2 Algs, 3 Runs).
- The agent must verify the statistical normality and variance of the Micro-Audit before proceeding to full scale.

## 5. Fault Tolerance & Persistence
**"Save-as-you-go."**
- Data loss is unacceptable. All experiments must save state incrementally.
- Filenames must be unique and descriptive, and metadata (durations, seeds) must be recorded for every run.
- Resumability must be verified manually by simulating an interruption (`KeyboardInterrupt`) during the Micro-Audit phase.

## 6. Metric Terminology & Discretization Rigor
**"Precision in Definition."**
- Avoid absolute terms like "Optimal" or "Theoretical Max" when referring to sampled reference sets. Use **"Sampled Reference"** or **"GT Baseline"**.
- Ground Truth density must be adjusted based on the manifold dimensionality:
    - **1D Manifolds (Curves)**: Require high-density (e.g., 10k points) to avoid discretization artifacts.
    - **2D+ Manifolds (Surfaces)**: Require standard-density (e.g., 2k points).
- Every metric report must explicitly mention the sampling nature of the reference set to prevent misinterpretation of "Performance Saturation" (HV > 100%).
*Failure to comply with these rules constitutes a regression in scientific quality and must be addressed immediately.*
