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

Imagine that we want to compare an optimization algorithm, such as
NSGA-III, on a problem like DTLZ2. We run the algorithm, obtain a set of
solutions and use IGD+ to measure how close they are to a reference
Pareto front. The program finishes without errors and prints a number.
Can we trust it?

There are several ways this result could be misleading. IGD+ might
have been calculated incorrectly; the experiment might have connected
otherwise correct components incorrectly; a software update might have
changed the solutions produced for a fixed seed; or the algorithm might
still reproduce that seed while becoming worse on other seeds.

Each possibility calls for a different kind of test. MoeaBench therefore
uses **unit tests, integration tests, numerical stability tests and
performance stability tests**. They answer related questions, but one
cannot stand in for another.

### 1.1 Unit tests — can we trust each piece?

Start with IGD+. Before involving an evolutionary algorithm, we can
construct two tiny sets of objective vectors and calculate their
expected distance by hand. A unit test supplies those inputs to the
indicator and checks its answer against our independent calculation.

The same principle applies to other components: a unit test can check
the evaluation of a MOP, how seeds are allocated, or how a population
is stored. These tests are usually inexpensive, so they run frequently.

If the IGD+ implementation is wrong, even an excellent MOEA will
receive a misleading score. Verifying the pieces first makes later
experimental results easier to trust and failures easier to diagnose.

### 1.2 Integration tests — can the pieces work together?

Next, we put the pieces together. We construct a MOP, configure an
MOEA, run a small experiment and compute an indicator from its output.
We may then save the experiment and load it again.

An integration test checks the entire path, including whether the seed,
configuration, objective values and provenance are passed correctly.
An indicator and a persistence module can each pass their own unit
tests while the experiment still loses essential information between
them. Integration tests detect problems at these boundaries.

### 1.3 Numerical stability tests — does the same experiment still give the same result?

Now imagine that a previous version of MoeaBench produced a final
front that we inspected and accepted. We record the experiment's
configuration, seed and expected output. Later, after changing the
software or its dependencies, we repeat **that same experiment**.

A numerical stability test compares the new result with the recorded
one, using the test's explicit numerical tolerance. We also protect
indicator calculations on fixed objective fronts with independently
derived reference values. Thus we can detect a changed optimization
trajectory separately from a changed metric calculation.

Why does this matter if the seed is fixed? A seed controls the random
choices, but numerical calculations also depend on algorithm code and
libraries. Small floating-point differences may alter selection and
produce different populations after several generations. With an
otherwise controlled environment, one execution is enough to check
this specific fixed-seed replay.

**A mismatch is a signal to investigate.** It might expose a genuine
regression, or it might come from a legitimate, reviewed change in the
numerical environment. The example below shows how to distinguish
these possibilities.

### 1.4 Performance stability tests — is the algorithm still as good in general?

One successful fixed-seed replay says little about the algorithm's
usual quality. Evolutionary algorithms use randomness: two seeds can
lead to different solutions, and a change may improve some runs while
worsening others.

To investigate typical performance, we compare a carefully chosen
set of MOP–MOEA experiments across **several paired seeds**. For each
seed we measure the same quality indicator, such as IGD+, in the
historical version and in the candidate. Statistical analysis then
determines whether the evidence supports an acceptably small
degradation.

This test is more expensive because it repeats actual optimization
runs. We use it explicitly for performance certification rather than
after every edit. Its interpretation depends on a reviewed historical
sample, a declared margin and enough statistical evidence.

### 1.5 What should we do when a numerical test fails?

Consider a simple story. Yesterday, a fixed-seed experiment produced
a particular set of solutions. Today, after updating MoeaBench or a
numerical library, we repeat exactly the same experiment and obtain
a different set. The numerical stability test fails because it can
no longer reproduce its historical reference.

**The failure tells us that the output changed, but not why.**
There are two important possibilities.

**A regression bug.** Perhaps a refactoring introduced an error in
the solver, changed a seed unexpectedly or corrupted an intermediate
result. In this case we find the defect, fix it and rerun the test.
The previously accepted result remains the correct expectation.
Replacing the reference with the new, incorrect result would only
hide the bug.

**A legitimate numerical change.** Perhaps a newer version of NumPy,
pymoo or the underlying math libraries calculates a few intermediate
values differently. Even tiny rounding differences can send an
evolutionary search along a different trajectory. The new result may
be perfectly valid under the new environment, but we must establish
that through investigation—by comparing environments, checking
independent mathematical contracts and understanding where the
trajectories first diverge. A library upgrade is not, by itself,
proof that the difference is harmless.

If the difference is understood and **explicitly accepted**, we can
establish a **new, reviewed and versioned numerical reference for
future tests**. We preserve the original reference and the evidence
behind it in the project's Git history and audit records. The old
result remains meaningful for the environment in which it was
validated; the new reference describes the newly accepted environment
or implementation. This is a deliberate change of the verification
contract, never an automatic response to a failing test. Under the
current MoeaBench policy, any modification to historical regression
tests or reference values requires advance maintainer approval and
must respect the applicable ADRs. The existing certification commands
do not automatically generate replacement numerical baselines.

There is a further subtlety: **a changed result need not mean worse
performance**. The solutions could differ while their IGD+ scores
over many seeds remain statistically acceptable. Conversely, one
seed could replay exactly while the algorithm performs worse on
other seeds. This is why numerical and performance stability are
tested separately. A good performance comparison does not excuse an
unresolved numerical failure: formal performance certification
requires a successful numerical certificate for the exact candidate
revision.

We encountered a real example with **NSGA-III × DPF1**. Matching
package versions did not always reproduce the historical front.
Investigation identified differences in NumPy and OpenBLAS execution
kernels. By controlling the reference runtime, the original
historical front was reproduced **without changing the algorithm or
its baseline**. A future, scientifically justified baseline revision
is possible under the review policy above, but this particular
incident did not require one.

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

When a numerical replay fails, **first investigate the difference**:
compare the source code, MOP/MOEA behavior, seeds, dependency versions
and native CPU/BLAS dispatch. A coding regression calls for a fix while
retaining the old reference. A legitimate numerical change can instead
justify a **new, reviewed versioned reference for future verification**,
provided its scientific contracts remain sound. The historical result
and its provenance must remain recoverable; our repository retains
earlier fixtures in Git history and the audit trail.

Current certification does **not** automatically rewrite a frozen
numerical fixture or generate a new version's reference fronts. A
deliberate reference revision needs independent justification,
maintainer approval before changing regression tests or expected
values, and an ADR review if it changes an architectural contract.
The historical NSGA-III/DPF1 case required only a controlled
numerical runtime to reproduce the original reference.
The read-only investigation report and replay scripts are archived in
[the NSGA-III/DPF1 numerical audit](audits/2026-11-18-nsga3-dpf1-environment-divergence.md)
and `scripts/audit/probe_*` for issue #92.

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

The word *baseline* is used for two related but distinct things.
A **numerical regression reference** describes an accepted output
for fixed inputs and a controlled numerical environment; an approved
scientific or environmental change can motivate a new versioned
reference for future tests, while preserving the previous one in
Git history. A **statistical performance baseline** is a historical
sample of results over many seeds, against which later releases
are compared. Creating either kind of new reference requires a
separate review; passing or failing a test cannot create one.

The command below concerns **only the historical statistical
performance sample**:

~~~bash
python test.py --record-performance-baseline --output NEW_REFERENCE.json
~~~

Today it is restricted to the exact certified historical **0.18.0**
source revision; that reference has already been generated and
approved. It is not a general-purpose command for updating numerical
regression expectations or recording arbitrary future performance
baselines. Supporting a new historical performance reference will
require an explicitly approved protocol and implementation change.
Neither `--certify-numeric` nor `--certify-performance`
creates or renews one.

### 3.5 What to do when a check fails

| Failure | Next action |
|---|---|
| Unit | Compare the local contract with an independent oracle or invariant. |
| Integration | Trace the boundary between components and their data/provenance. |
| Numerical stability | Preserve the frozen expectation; inspect seeds, code, versions and effective native kernels. |
| Performance certification: `REGRESSION_DETECTED` | Investigate the paired quality change and its algorithmic cause. |
| Performance certification: `INSUFFICIENT_EVIDENCE` or `INSUFFICIENT_POWER` | Keep the negative release decision; review variability and study adequacy without selecting favorable seeds or changing the margin after seeing results. |
| Numeric certification | Fix any failing basic check before recording a certificate. |

**A failed test starts an investigation, not an automatic
rebaseline.** Fix regressions against the accepted reference.
If a numerical difference is legitimate and a new baseline is
scientifically justified, obtain approval first, record the
new reference and retain the historical one. The numerical and
performance gates remain separate; a failed numerical certification
prevents formal performance certification of that candidate.

For complete implementation-level details, see
[environment verification](environment-verification.md),
[performance certification](performance-stability.md),
[ADR 0050](adr/0050-certified-environment-verification-and-provenance.md)
and [ADR 0056](adr/0056-versioned-certification-and-performance-reference-lifecycle.md).
