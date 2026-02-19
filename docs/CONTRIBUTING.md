<!--
SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# Contributing to MoeaBench

Thank you for your interest in MoeaBench! This document outlines the process for contributing to the framework and the coding standards we maintain to ensure quality and consistency.

## 1. Contributor Workflow

To ensure a smooth collaboration, we follow a structured workflow for all contributions:

1.  **Open an Issue**: Before implementing any significant changes, please open an issue in the [official repository](https://github.com/monacofj/moeabench) to discuss your proposal.
2.  **Fork**: Fork the repository if you haven't already.
3.  **Branching**: Create a new branch with the following naming pattern:
    *   `feature|fix/<issue-number>/<short-description>`
    *   Example: `feature/42/add-new-metric`
5.  **Implement**: Write your code adhering to our [Code Conventions](#2-code-conventions) and [Architectural Principles](design.md).
6.  **Verify (Mandatory)**: Ensure your changes satisfy the automated test suite. See [Verification and Testing](#3-verification-and-testing) for details.
7.  **Pull Request**: Submit a Pull Request targeting the `main` branch. Briefly describe your changes and link the relevant issue.

---

## 2. Code Conventions

We strive for high-quality, "Pythonic" code that is efficient and easy to maintain.

### Style and Formatting
- **PEP 8**: We strictly follow the PEP 8 style guide. Use tools like `black` or `ruff` to ensure consistent formatting.
- **Pythonicness**: Avoid verbose, non-idiomatic code. Use list comprehensions, decorators, and context managers where appropriate.

### Type Hints
- **Mandatory Type Hints**: All public functions and classes MUST include type hints. 
- Use the `typing` module (or built-in types for Python 3.10+) to ensure clarity and enable better static analysis for users.

### Documentation (Docstrings)
- **Standard Format**: Use the NumPy/Google docstring format.
- **Clarity**: Docstrings should explain parameters, return values, and any exceptions raised. 
- **The Analytical Tone**: Since MoeaBench is an analysis-driven tool, avoid overly cryptic language. Explain the "why" in complex functions.

### Performance and Data Handling
- **Vectorization**: This is our core technical style. Never iterate over populations in Python loops. Use **NumPy** vectorized operations exclusively.
- **Smart Containers**: Use `SmartArray` for numerical data to ensure label consistency and easy plotting integration.

### Developer Documentation
Before contributing, please read the following technical resources to understand our core principles and evolution:
- **[Design Philosophy](design.md)**: Detailed explanation of our **Technical Storytelling** (Scientific Narrative) style, the Rich Result architecture, the `fooplot` convention, and mandatory vectorization.
- **[Architecture Decision Records (ADRs)](adr/)**: Registry of the fundamental design choices that shaped the framework.

---

## 3. Verification and Testing

We maintain a rigorous testing pyramid using `pytest` and a central orchestrator. No contribution will be accepted if it breaks the core logic or mathematical invariants.

### Test Orchestrator
Use the `test.py` script in the root directory to run the test suite. 

**Always run the base tests before committing:**
```bash
./test.py  # Runs Unit Tests + Light Tier
```

### Testing Tiers
- **Unit Tests (`--unit`)**: Functional validation of core classes (`Population`, `Experiment`), metrics, and persistence.
- **Light Tier (`--light`)**: Mathematical invariants of benchmarks (e.g., DTLZ/DPF geometry) without stochastic execution.
- **Smoke Tier (`--smoke`)**: Regression check of algorithm convergence against the v0.8.0 release baselines (IGD thresholds).
- **Heavy Tier (`--heavy`)**: Full statistical calibration (N=30) and hypothesis testing for large-scale performance verification.

### Contribution Rules for Tests
- If you add a new benchmark, you MUST add corresponding tests in `tests/test_light_tier.py`.
- If you modify core logic, ensure `tests/unit/` remains at 100% pass rate.
- Avoid adding tests that depend on heavy CPU usage or GUI backends.

### Calibration and Baselines
Maintaining scientific consistency is paramount. The framework relies on an **Oracle Baseline** (`tests/baselines_v0.8.0.csv`) and a **Calibration Report** (`tests/CALIBRATION_v0.8.0.html`).
- If your contribution affects the core IGD/HV performance, you MUST regenerate the baselines and the report.
- Raw calibration traces are stored in `tests/calibration_data/`.
- Use `tests/calibration/compute_baselines.py` and `tests/calibration/generate_visual_report.py` to maintain these artifacts.

---

## 4. AI Usage Disclosure

While we don't rule out the use of auxiliary AI assistance tools in this project, contributors must explicitly disclose such usage, take full responsibility for the intellectual work, and ensure all submissions strictly adhere to our performance requirements and licensing compatibility.

---

## 5. Communication

Feel free to contact the authors for questions, collaboration ideas, or architectural discussions. See the [AUTHORS](../AUTHORS) file for contact details.
