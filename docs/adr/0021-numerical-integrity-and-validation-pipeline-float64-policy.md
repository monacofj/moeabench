<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# ADR 0021: Numerical Integrity and Validation Pipeline (Float64 Policy)

## Status

Accepted (Implemented in v0.7.6)

## Context

With MoeaBench measuring performance differences at the scale of $10^{-4}$ or $10^{-5}$, the risk of floating-point truncation (using `float32`) became a critical concern for scientific validity. It was necessary to certify that no part of the ingestion or computation pipeline downcasts data.

## Technical Decision

We established a **Strict Numerical Integrity Policy**:

1.  **Float64 Standard**: All internal matrices (objectives, variables) and intermediate metric calculations must use **64-bit double precision** (`float64`).
2.  **Explicit Audit**: Every major calibration cycle must include a "Data Type Audit" (`audit_precision.py`) to verify that `pd.read_csv` and `numpy` stack operations preserve the 64-bit width.
3.  **Precision Certification**: The final Peer Review Response or Calibration Report must include a "Precision Certification" note verifying the audit's success.

## Consequences

### Positive
*   **Scientific Reliability**: Ensures that IGD/HV differences are mathematically stable and not artifacts of quantization.
*   **Fault Tolerance**: Prevents catastrophic precision loss in high-dimensional or long-running evolutionary trajectories.

### Negative
*   **Memory Footprint**: Doubling the bit-width increases memory usage, though this is negligible for current population sizes (e.g., $N=200$).
