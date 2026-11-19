<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# Testing MoeaBench: a practical tutorial

MoeaBench distinguishes **correctness**, **integration**, **numerical
reproducibility**, and **stochastic performance**. Each requires a different
kind of evidence. This tutorial first explains the four types of tests, then
shows which commands to run, and finally explains the protocols used to
interpret and certify the results.

## 1. Why four types of tests?

Imagine an experiment that runs NSGA-III on DTLZ2 and evaluates its final
population using IGD+. Before trusting the reported quality, we need to
answer four questions.

### 1.1 Unit tests — is each component correct?

**Purpose.** Verify an isolated mathematical or software contract: for
example, the IGD+ formula, a MOP's evaluation function, the allocation of
random seeds, or how a population is stored.

**How.** Supply small, controlled inputs and check results against an
independent calculation or an explicit invariant. A test of IGD+ might use
two hand-constructed objective fronts whose distance can be derived without
running a MOEA.

**Why it matters.** A complete optimization experiment can execute
successfully while an indicator silently reports the wrong number. Unit
tests detect that kind of error at its source. They do not establish that
all components cooperate correctly.

### 1.2 Integration tests — do the components work together?

**Purpose.** Verify realistic workflows across component boundaries.

**How.** Construct a MOP and MOEA, execute a small experiment, inspect the
result, calculate an indicator, and, where applicable, save and reload it.
The test checks that data, configuration and provenance pass correctly
between APIs.

**Why it matters.** Correct individual components can still interact
incorrectly—for example, a saved experiment might lose its seed or
normalization configuration.

### 1.3 Numerical stability tests — can we reproduce a known result?

**Purpose.** Detect changes in an already validated numerical result.

**How.** Replay a prescribed MOP–MOEA configuration with a fixed seed,
population size, generation budget and numerical environment; compare its
final front with a **frozen reference**. Separate numerical tests feed
fixed fronts to indicators and check their values against independently
justified expected numbers. Each comparison has a declared tolerance;
some also require the exact shape of the front.

**Why it matters.** Changes in the solver, random-number handling, library
versions or CPU/BLAS kernels can alter a trajectory even when unit and
integration tests pass. **One run is sufficient to check one fixed-seed
replay** in the specified environment; repeating the same seed does not
characterize performance across random seeds.

### 1.4 Performance stability tests — is stochastic quality still acceptable?

**Purpose.** Determine whether a new version has materially worsened the
quality typically obtained by an algorithm.

**How.** Execute the **same preselected MOP–MOEA cases with paired seeds**
on a candidate version, compare their IGD+ scores with an approved
historical sample, and use a prespecified statistical decision rule.
These tests use several seeds because different runs can have different
qualities.

**Why it matters.** A solver can reproduce its one frozen trajectory and
nevertheless perform worse on other seeds. Performance testing detects a
distribution-level degradation that deterministic replay alone cannot
establish. It is computationally more expensive and is invoked explicitly,
not after every code change.

### 1.5 Example: numerical replay fails, but performance may be unchanged

Suppose an earlier version produces a particular three-point final front
for `NSGA3 × DPF1` with seed `12345`. A later execution, using a different
native floating-point kernel, produces a two-point front. The **numerical
stability test fails**, since that frozen output was not reproduced.

Now consider a *separate, hypothetical* experiment across many paired
seeds: the new fronts differ, but their mean IGD+ degradation remains
below the predeclared tolerance, with enough statistical evidence. In
that case, **performance stability can pass despite a different
deterministic trajectory**. The opposite is also possible: one frozen
seed reproduces exactly, while many other seeds perform worse, causing
the performance test to fail.

The distinction is scientific, not permission to ignore a failing test.
The historical NSGA-III/DPF1 discrepancy was actually traced to NumPy and
OpenBLAS dispatch and resolved for the controlled reference environment
without rewriting the frozen front. The *performance-pass scenario above
is illustrative*, not a claim that a historical performance comparison
was run for that discrepancy. **For release certification, MoeaBench
requires numerical certification of the exact candidate revision before
it permits performance certification.**

## 2. Which command should I run?

Execute commands from the **repository root**, in an environment where
MoeaBench and its test dependencies are installed. The normal commands
below are read-only with respect to historical certificates and baselines.

### 2.1 During ordinary development

**To verify the implementation of individual components:**

~~~bash
python test.py --unit
~~~

This runs **all** unit tests. The `--smoke` or `--basic` level does not
reduce the unit scope.

**To verify that components work together:**

~~~bash
python test.py --integration --smoke
~~~

For the complete integration suite, use:

~~~bash
python test.py --integration --basic
~~~

The complete integration suite needs two example archives on a clean
checkout; their preparation is shown in
[Section 2.3](#23-prepare-the-integration-examples).

**To check a small set of deterministic numerical regressions:**

~~~bash
python test.py --numeric-stability --smoke
~~~

**To check ordinary code changes in one command:**

~~~bash
python test.py
~~~

This runs all unit tests plus integration and numerical **smoke**
checks. It does not run the expensive statistical performance study.

**To see which tests are classified in each scope:**

~~~bash
python test.py --list
python test.py --list --numeric-stability --basic
~~~

`--list` filters the displayed classifications; `--basic` during
execution includes the smoke and basic levels. For integration and
numerical stability, `--deep` is not an active reference level.

### 2.2 Before full numerical verification: select the reference runtime

The complete frozen numerical suite is sensitive to native NumPy and
OpenBLAS kernels. For the validated **Linux/x86_64 reference profile**,
use the pinned numerical dependencies described in
[Environment verification](environment-verification.md), then run:

~~~bash
export NPY_DISABLE_CPU_FEATURES=X86_V4
export OPENBLAS_CORETYPE=HASWELL
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python scripts/check_numerical_reference_runtime.py
python test.py --numeric-stability --basic
~~~

The preflight checks the **effective** NumPy and OpenBLAS kernels and
thread counts, not just environment variable values. A failing
preflight means the current host cannot claim the validated reference
profile. These settings are for controlled verification, not a general
requirement for users running optimization experiments.

### 2.3 Prepare the integration examples

On a clean checkout, create the archives required by the full
integration suite. This reproduces the deterministic setup used in CI:

~~~bash
python - <<'PY'
from pathlib import Path
import moeabench as mb

for algorithm, filename in (
    (mb.moeas.NSGA2, "example_full_exp1.zip"),
    (mb.moeas.NSGA3, "example_full_exp2.zip"),
):
    path = Path("examples") / filename
    if not path.exists():
        exp = mb.experiment(
            mop=mb.mops.DTLZ2(M=3),
            moea=algorithm(population=32, generations=8, seed=17),
        )
        exp.run(repeat=3, silent=True)
        exp.save(str(path))
PY
~~~

**To run all unit, integration and numerical basic tests**, first complete
Sections 2.2 and 2.3, then execute:

~~~bash
python test.py --basic
~~~

### 2.4 Certify a candidate version

Certification is a deliberate release operation. **First** assign the
candidate its new package version, use the approved runtime, and prepare
the integration examples. **Then certify its numerical contracts:**

~~~bash
python test.py --certify-numeric
~~~

This reruns the complete `basic` suite, refuses filtered/incomplete
verification, and only on success records a compact numerical
certificate for the exact version, source revision, environment and
frozen-reference hashes. An existing certificate for a different source
revision cannot be silently replaced.

**To certify stochastic performance**, first ensure that the same
candidate revision has a valid numerical certificate; then compare it
with the approved historical reference. For a *future* 1.0.0 candidate:

~~~bash
python test.py --certify-performance \
  --reference moeabench/resources/certifications/0.18.0/performance-reference.json \
  --output moeabench/resources/certifications/1.0.0/performance-vs-0.18.0.json
~~~

The output is a `CERTIFIED` or `NOT_CERTIFIED` verdict with diagnostics.
The reference stays unchanged. This is an example for a future
candidate: **1.0.0 has not been certified or published by these
instructions**.

**To inspect previously recorded numeric certificates:**

~~~bash
python -c "import moeabench as mb; print(mb.system.certification_history())"
~~~

A dependency-version match can be inspected using
`mb.system.matching_certified_environments()`, but a version match by
itself is not evidence of numerical or statistical equivalence.

## 3. How do the verification protocols work?

### 3.1 Routine tests and CI

The test runner has three ordinary pytest scopes: `unit`,
`integration` and `stability` (invoked as
`--numeric-stability`). Performance certification is the **fourth
verification type** but runs as a separate experimental command,
outside the pytest scope/level matrix.

- **Smoke**: a small sentinel set for integration and numerical
  reproducibility.
- **Basic**: smoke plus the complete integration and frozen numerical
  checks; all unit tests always run.
- **Deep**: historical, without an available statistical baseline;
  it is not a release gate and does not substitute for the explicit
  performance certification command.

Two GitHub Actions workflows exercise the ordinary checks:
[unit contract regression checks](../.github/workflows/unit-contract-tests.yml)
and [stability verification separation](../.github/workflows/stability-separation.yml).
Both include a controlled reference-numeric job. Unit tests also
exercise the *logic* of the statistical protocol using synthetic
values; these do not run the actual multi-seed performance experiment.

### 3.2 Frozen numerical references

Numerical tests protect two separate things: (a) indicator calculations
on fixed objective fronts, with analytically or independently derived
expected values; and (b) fixed-seed solver trajectories, reference
front geometry and PRNG behavior.

The canonical fixtures are:

- `tests/conventional_metric_reference_v1.json` — GD, IGD, GD+,
  IGD+, EMD and HV examples;
- `tests/stability_reference_fronts.json` — historical solver fronts;
- `tests/stability_reference_gt.json` and
  `tests/stability_gt_geometry_reference.json` — GT/reference geometry.

Different checks have different tolerances. For example, a frozen
solver-front check requires the expected shape and uses
`np.allclose(atol=1e-12, rtol=1e-8)`; independent metric values can use
`atol=1e-10, rtol=0`. Reproducibility is judged against each test's
explicit contract.

**GT calibration is a separate MOP responsibility.** The public
`mop.calibrate()` establishes MOP-owned Ground Truth; it is not a
performance baseline and is never regenerated by certification.
For example, absolute HV uses calibrated MOP GT as its denominator,
while an explicit `ref` specifies normalization geometry
([ADRs 0019, 0035 and 0047](adr/README.md)).

When a numerical replay fails, preserve the fixture and investigate
the first differing operation: source change, MOP/MOEA behavior,
component seed, dependency version or native CPU/BLAS dispatch.
The documented NSGA-III/DPF1 case illustrates why matching software
version numbers was insufficient. **Changing a historical reference
or regression test requires prior explicit approval**, and any
ADR-incompatible change must be authorized first.

### 3.3 Historical performance reference and statistical decision

The approved inaugural reference is **MoeaBench 0.18.0**, recorded
from its numerically certified original source revision. It contains
**384 actual solver runs**: 64 fixed seeds for each of six cases:

| Problem | Algorithms |
|---|---|
| DTLZ2 (3 objectives) | NSGA-II, NSGA-III, MOEA/D |
| DTLZ7 (3 objectives) | NSGA-II, NSGA-III |
| DPF1 (M=3, D=2) | NSGA-III |

Each case uses population 48 and 60 generations. The historical sample
stores IGD+ as the decisive indicator, plus GD+ and EMD for diagnosis,
using frozen 512-point objective references. Candidate measurements
reuse those exact objective references and the same paired seeds.

For each case, eight pairs are used as an **independent pilot** to
estimate variability of candidate-minus-reference differences.
The pilot is excluded from the final interval. The protocol then plans
8–56 fresh confirmatory pairs, targeting 95% model-based probability
of establishing noninferiority if the true mean change is zero. If that
power target cannot be met within the budget, certification fails
with `INSUFFICIENT_POWER`.

The allowable mean degradation is **0.1% of the historical mean IGD+**
for that case (with a numerical floor of `1e-12`). Simultaneous
Bonferroni-adjusted Student-t intervals control familywise error at
**1%** across the six cases. A candidate is `CERTIFIED` only when
**all six upper confidence bounds** are below their declared margins.
A statistically demonstrated regression, an inconclusive comparison
or inadequate power yields `NOT_CERTIFIED` with a specific reason.
The result is an operational release decision, not a universal
equivalence claim.

The reference was independently reviewed by an
[automated technical audit](https://github.com/monacofj/moeabench/actions/runs/44409693329)
that verified the recorded observations, hashes, provenance and
18 independent solver replays. Approval does not mean that every
observation received manual human review.

### 3.4 Creating a new baseline versus certifying a release

A numerical certificate records a version, source revision,
dependencies, effective runtime and **hashes of canonical numerical
fixtures**; it does not duplicate solver populations or recalculate
GT. Performance references, by contrast, store the actual frozen
historical experimental sample. These records live in
`moeabench/resources/certifications/<version>/`.

Recording a baseline is an **exceptional, independently approved**
operation. Its command is:

~~~bash
python test.py --record-performance-baseline --output NEW_REFERENCE.json
~~~

The present implementation restricts inaugural reference creation to
the exact certified historical **0.18.0 source revision**; that
reference has already been created and approved. Do not rerun the
operation against the existing file. Future baseline policy changes
require review. Neither `--certify-numeric` nor
`--certify-performance` creates or renews a baseline.

### 3.5 What to do when a check fails

| Failure | Next action |
|---|---|
| Unit | Compare the local contract with an independent oracle or invariant. |
| Integration | Trace the boundary between components and their data/provenance. |
| Numerical stability | Preserve the frozen expectation; inspect seeds, code, versions and effective native kernels. |
| Performance certification: `REGRESSION_DETECTED` | Investigate the paired quality change and its algorithmic cause. |
| Performance certification: `INSUFFICIENT_EVIDENCE` or `INSUFFICIENT_POWER` | Keep the negative release decision; review variability and study adequacy without selecting favorable seeds or changing the margin after seeing results. |
| Numeric certification | Fix any failing basic check before recording a certificate. |

**No failed check authorizes rebaselining.** The numerical and
performance gates are independently assessed and preserve their
historical references. A failed numerical certification prevents
starting the formal performance certification of that candidate.

For complete implementation-level details, see
[environment verification](environment-verification.md),
[performance certification](performance-stability.md),
[ADR 0050](adr/0050-certified-environment-verification-and-provenance.md)
and [ADR 0056](adr/0056-versioned-certification-and-performance-reference-lifecycle.md).
