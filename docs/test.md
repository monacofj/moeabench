<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# Testing, numerical reproducibility, and performance verification

MoeaBench supports scientific experiments in which both the **correctness of a
measurement** and the **reproducibility of an optimization result** matter.
The testing system exists to tell these questions apart, expose a failure at
the appropriate level, and retain the evidence needed to understand it.

This manual explains the reasoning behind the tests, what we learn from a
pass or failure, how references are established and reviewed, and how a
release environment is certified. The command reference and current
implementation details follow the scientific explanation.

## 1. Why the tests have several layers

Consider a study that uses a multiobjective problem (MOP), an evolutionary
algorithm (MOEA), and an indicator such as IGD.

We want to be able to trust four successive claims:

**First, the indicator computes the intended quantity.** Given a small
front for which we can derive IGD independently, the implementation must
return the correct result. A test with this purpose does not need to
execute any evolutionary algorithm.

**Second, the components work together.** Constructing the MOP and MOEA,
running an experiment, collecting a population, computing an indicator,
saving the result and loading it again must respect the public contracts.
Correct pieces can still interact incorrectly.

**Third, a previously validated numerical experiment can be replayed.**
With a specified MOP, MOEA, seed, evaluation budget, numerical environment
and frozen expected front, a repeat execution should reproduce the recorded
result according to the test's declared tolerance. We can also test the
indicator on a *frozen* objective-space front, independently of the
algorithm that generated it.

**Fourth, the typical performance of a stochastic algorithm remains
acceptable.** One repeatable execution does not tell us whether the
distribution of quality across independently seeded runs has deteriorated.
That question requires a designed experiment, repeated observations, a
reviewed historical reference, and an explicit statistical decision rule.

These claims motivate the distinct verification activities below.

| Verification | Scientific question | Evidence | Typical cost |
|---|---|---|---|
| Unit tests | Does this component satisfy its local contract? | Independently derived values, invariants, small controlled cases | Low |
| Integration tests | Do the components compose correctly? | Representative end-to-end workflows | Moderate |
| Numerical stability | Can the approved numerical result be replayed? | Fixed inputs and frozen metric outputs; seeded MOP–MOEA fronts | From small to moderate |
| Performance stability | Has mean stochastic quality degraded materially? | Repeated paired runs against an approved *prior* reference | Potentially high |
| Environment certification | Which dependency versions passed the required development suite? | Successful complete `basic` run followed by a version manifest | Cost of complete `basic` suite |

The last two are especially easy to confuse: **environment certification does
not manufacture an experimental performance baseline**, and a statistical
test does not certify dependency versions.

## 2. The normal verification sequence

A developer should normally proceed in the order in which failures can be
interpreted meaningfully.

### 2.1 Verify local contracts first

Run the unit tests. These protect the public API, data types, metric
semantics, mathematical invariants, GT calibration, float64 boundaries,
persistence, seed allocation and other narrow contracts.

For example, a unit test of GD or IGD supplies a tiny pair of objective
sets and compares the returned value to an independently calculated
distance. If it fails, the indicator or its assumptions must be examined
before an expensive MOEA experiment can be trusted.

A unit test that passes provides evidence about its **specific inputs and
contract**. It cannot establish correctness for every possible input.

### 2.2 Verify complete workflows

Integration tests exercise actual combinations of the API: an experiment
with a MOP and a MOEA, analysis of its result, view dispatch, saving and
loading, and the provenance retained by persistence.

An integration failure means a workflow contract is not respected, even if
the individual functions pass their unit tests. Debugging should start
with the boundary between the components involved.

### 2.3 Replay the approved numerical references

Numerical stability has **two independent parts**.

The *indicator* part fixes an objective-space front and a geometric
reference, then checks the calculated indicator against a stored,
independently justified value. This catches changes in the measurement
even when the optimization algorithm is unchanged.

The *algorithm* part fixes a MOP, a MOEA, seeds, a population size and a
generation budget, then compares the produced front with the archived
front, including its shape and coordinates. This catches changes in a
deterministic trajectory caused by algorithm code, RNG behavior or
underlying numerical dependencies, even when the metrics themselves
remain correct.

Under a truly fixed execution environment and seed, **one run is
sufficient for this particular replay comparison**. Repeating it many
times does not provide new evidence about a stochastic performance
distribution. It is useful to repeat a failing run for diagnosis, but
the essential contract remains a fixed input-to-output comparison.

The numerical references are preserved across releases. Their failure
is important information, not a reason to update the expected values.

### 2.4 Compare stochastic performance separately

A fixed seed can replay perfectly while an algorithm's average
performance over different seeds becomes worse. Conversely, a different
numerical trajectory may not imply any relevant change in the quality
distribution.

A performance comparison starts with a **scientific design**: choose a
small, representative set of MOP–MOEA configurations, the quality metric,
a fixed ground-truth or reference front, a common list of seeds, a
meaningful degradation margin, and enough paired repetitions to support
an inference. This is not an automatic grid of every available
combination, nor does every scenario automatically require 30 runs.

The previous version's measured quality for each of those seeds
constitutes an *external performance reference*, after independent
review. The candidate version then runs the **same declared cases and
seeds** and compares the observations.

This step is **optional and explicitly invoked**. A numerical stability
failure does not trigger it. This separation lets us first understand
whether the candidate reproduces the required numerical contracts
before drawing conclusions about experimental performance.

## 3. What happens when something fails?

The following is the operational decision process. A failing check must
remain visible until its meaning is established.

| Observation | First action | What is not justified |
|---|---|---|
| Unit test fails | Check the local contract and independently derived oracle | Changing the expected value merely to pass |
| Integration fails | Trace the workflow and interface boundary | Assuming isolated component tests prove the workflow works |
| Numerical replay fails | Keep the frozen reference; compare seeds, implementation, versions, CPU/BLAS profile and the earliest differing operation | Discarding or automatically regenerating the historic front |
| Performance assessment reports `FAILED` | Examine baseline provenance, paired observations, effect size, assumptions and real algorithmic changes | Editing the prior performance values |
| Performance assessment reports `INCONCLUSIVE` | Preserve the uncertainty; review statistical power, pilot variability and study design | Calling the result a pass or an equivalence proof |
| `--certify-numeric` encounters a failing `basic` batch | Refuse certification; investigate the original failure | Updating the certified manifest anyway |

### 3.1 A numerical failure is a diagnosis, not a recalibration order

Numerical tests may fail for several reasons. The implementation may
have regressed, an accepted numerical contract may have been
intentionally changed, a seed may be allocated differently, or a
native floating-point library may select a different CPU kernel.

We first reproduce the failure, identify the **first** differing
operation, and determine whether it is a genuine code error or a
runtime difference. Mathematical unit oracles and ADRs guide that
assessment. Where necessary, the investigation is tracked in its own
issue. A change contrary to an accepted ADR must be presented to the
maintainer and authorized before implementation.

The NSGA-III × DPF1 case illustrates why. The historical front differed
across CI machines with matching Python/package versions. Controlled
experiments isolated NumPy `power` SIMD dispatch during crossover and
OpenBLAS kernel choice during NSGA-III survival. Pinning those two
environment properties restored the **original** historical front;
neither the solver nor the regression test was edited.

See [issue #92](https://github.com/monacofj/moeabench/issues/92),
[ADR 0050](adr/0050-certified-environment-verification-and-provenance.md),
and [Environment verification](environment-verification.md).
A passing test under that controlled profile does not imply universal
bitwise reproduction on arbitrary hardware.

### 3.2 When may a numerical reference change?

A historically accepted numerical reference is a **versioned scientific
contract**. It is retained by default.

If the implementation was defective, fix it and check against the
existing mathematical and regression contracts. If a *deliberate,
scientifically justified* change necessarily alters approved numerical
behavior, first identify the governing ADR, describe the change,
compare old and new results independently, and obtain explicit
maintainer authorization **before changing a historical regression
test, fixture or expected value**. Preserve the earlier reference's
provenance in the history and introduce the reviewed new reference
through a separate change.

A dependency upgrade that causes numerical drift does **not** by itself
justify replacing the baseline. First determine whether the designated
reference runtime can be reproduced and what changed.

### 3.3 Why passing a performance comparison is different

The performance gate has exactly two **operational** outcomes:
`CERTIFIED` and `NOT_CERTIFIED`. Every case must provide positive
statistical evidence of noninferiority to its immutable earlier
reference. A detected degradation or insufficient statistical precision
blocks certification and is recorded as a **diagnostic reason**, so the
maintainer knows whether to investigate an algorithmic change or the
study's uncertainty. A failure never refreshes the baseline.

The conclusion concerns the **paired mean IGD+ change** for six
predeclared MOP–MOEA configurations, not identical entire distributions,
all supported algorithms, or an unconditional guarantee of numerical
reproducibility. Those separate scientific claims require separate tests.

## 4. The different references — and what “calibration” means

Several files may be called a *reference*, *baseline* or *calibration*.
They serve different purposes and must be managed separately.

### 4.1 Ground Truth belongs to the MOP

The Ground Truth (GT) describes the optimization problem's target in
objective space. It may come from an analytical Pareto front, an
explicit reviewed data file or an empirical search, depending on the
MOP and the study. The core public `mop.calibrate()` operation
establishes a MOP-owned GT; `mop.calibrated_gt` exposes it.

The GT is necessary for some measurements. In particular,
`mb.metrics.hv(..., scale="abs")` uses the **calibrated MOP GT**
to determine the denominator. When an external `ref` is supplied,
that reference defines normalization geometry, **not** a replacement
GT. This is an important independent public contract.

Changing or recomputing a GT is a scientific/calibration decision.
It is independent of `--certify-numeric` and never triggered by it.

### 4.2 Numerical stability references are frozen observations

The two kinds of active numerical reference are:

- Fixed input fronts with independently derived expected indicator
  values (including exact distance and hypervolume examples).
- Historical seeded solver-front outputs and GT geometry signatures.

They support reproducibility and regression detection. In the current
repository, these include
`tests/conventional_metric_reference_v1.json`,
`tests/stability_reference_fronts.json`,
`tests/stability_reference_gt.json`, and
`tests/stability_gt_geometry_reference.json` (the unchanged geometry
oracle formerly embedded in the retired audit dataset).

The derivation of the 2D/3D conventional-metric values is recorded in
[the metric-reference audit](audits/conventional-metric-reference-v1.md).
The independent arithmetic verifier,
`python scripts/audit/verify_conventional_metric_references.py`,
checks those frozen values without calling the production numerical
implementation. Tests must not recompute and overwrite their own
expected values.

### 4.3 Performance references are historical *experimental samples*

The inaugural **0.18.0 candidate reference** was measured on the
previously certified 0.18.0 source
[at this exact historical revision](https://github.com/monacofj/moeabench/commit/0d3e66804351e23b93fb483ff25e178e06ef6e12).
The [384-run evidence](https://github.com/monacofj/moeabench/actions/runs/44401850675)
contains six predefined cases with 64 independent seeds per case,
one frozen 512-point objective-space reference per MOP, and IGD+,
GD+ and EMD observations. Source revision, reference-front checksums,
runtime versions, CPU/BLAS profile and all seed-wise results are
retained for scientific review.

A historical statistical reference is **never recomputed** when
numeric certification runs or when a future candidate fails.
`--record-performance-baseline` is an explicit, one-time operation.
It creates a candidate that must undergo independent integrity review
and explicit approval before use. Later candidates consume the frozen
reference and repeat the same seeds, rather than drawing a favorable
new reference. No MOEA solver fronts are duplicated in the numerical
certificate archive; these are separate scientific datasets.

The first reference's across-seed variance is not a substitute for
the *future paired-difference variance*. `--certify-performance`
obtains the latter from an independent eight-pair pilot and allocates
a bounded confirmatory sample. See
[Performance certification](performance-stability.md).

### 4.4 The certified environment manifest records versions

The manifest
[`moeabench/resources/certified_environment.json`](../moeabench/resources/certified_environment.json)
contains the *developer-certified software-version snapshot*: Python,
MoeaBench and the numerical dependencies named by the environment
verification implementation.

This is an identity record, **not** a list of expected indicator values,
a solver-front baseline or a statistical performance sample. It also
does not fully encode the effective native CPU/BLAS kernel selection.

The previously certified manifest identified MoeaBench **0.16.1**; the
0.18.0 candidate was subsequently certified through the full basic
suite, and its new manifest is retained in version control. Future
certifications replace the manifest only after the complete basic suite
succeeds. Git history can preserve earlier
manifest revisions. It would be misleading to discard frozen numerical
regressions merely because a new manifest was certified.

## 5. Operational test scopes, levels and exact commands

The top-level [`test.py`](../test.py) executes pytest in separate
subprocess batches. With no scope selected, it runs unit, integration
and numerical stability. The default level is `smoke`.

There are three **pytest scopes**, with a shared marker vocabulary:

| Scope | Purpose | Executed test reach |
|---|---|---|
| `unit` | Local correctness, public contracts, mathematical oracles | **All** unit tests, regardless of level flag |
| `integration` | Representative composition/workflows | Smoke or inclusive basic |
| `stability` | Fixed-input indicator and seeded numerical replay | Smoke or inclusive basic |

The explicitly requested `--certify-performance` command
is **outside this pytest scope/level matrix**. It requires an external
reference JSON file and does not trigger other scopes.

### 5.1 Smoke, basic and historical deep

`smoke` provides quick sentinel checks. For integration, it selects
the central representative workflows; for numerical stability, it
selects RNG signatures and a small set of GT and seeded-front checks.
The **unit scope is total-only**, so even `--unit --smoke` executes
all unit tests.

`basic` includes smoke and adds the full integration scope and the
more demanding numerical references. It is the required development
tier for environment certification.

The old `deep` statistical suite has an unavailable historical
baseline and is not an active verification or release gate.
`--deep` without a scope, `--numeric-stability --deep` and
`--integration --deep` are rejected. `--unit --deep` still
means all unit tests, **not** statistical performance validation.

### 5.2 Command reference

| Command | Purpose |
|---|---|
| `python test.py` | Full unit + integration smoke + numerical smoke |
| `python test.py --smoke` | Same default |
| `python test.py --basic` | Full unit + integration smoke/basic + numerical smoke/basic |
| `python test.py --unit` | All unit contracts |
| `python test.py --unit --smoke` / `--basic` / `--deep` | All unit contracts |
| `python test.py --integration --smoke` | Integration smoke only |
| `python test.py --integration --basic` | Integration smoke/basic |
| `python test.py --numeric-stability --smoke` | Numerical smoke only |
| `python test.py --numeric-stability --basic` | Fixed numerical smoke/basic |
| `python test.py --list` | Inspect classified tests |
| `python test.py --certify-numeric` | Complete basic suite and versioned, compact numerical certificate |
| `python test.py --record-performance-baseline --output FILE` | Explicit 0.18.0 historical performance sample, initially awaiting approval |
| `python test.py --certify-performance --reference FILE --output FILE` | Compare against an approved prior reference and write an immutable binary verdict |

Scope flags and level flags are mutually exclusive.
All three certification operations are **independent** of scope and
level selectors. The two performance operations require an explicit
artifact path; no historical baseline can be silently selected or
renewed. The previous provisional certification/statistical CLI names
have been removed before 1.0.0, with their original source and
decisions retained in Git history.

### 5.3 Classification and listing

[`tests/conftest.py`](../tests/conftest.py) assigns one scope and
one level marker per official test at collection time. The markers are:

```text
scope_unit  scope_integration  scope_stability
level_smoke  level_basic  level_deep
```

The actual runner selections are:

```text
unit:               scope_unit
integration smoke:  scope_integration and (level_smoke)
integration basic:  scope_integration and (level_smoke or level_basic)
stability smoke:    scope_stability and (level_smoke)
stability basic:    scope_stability and (level_smoke or level_basic)
```

Unit tests are collected from `tests/unit/` and
`tests/test_light_tier.py`. The unit smoke classification identifies
`test_api_surface.py`, `test_compare_aliases.py` and
`test_system.py`, but the runner **always executes every unit
test**. Integration files under `tests/integration/` are basic
unless explicitly designated smoke in `conftest.py`.

Numerical stability files form an **explicit allowlist**:

| File | Active classification |
|---|---|
| `tests/test_stability_smoke.py` | `scope_stability`, `level_smoke` |
| `tests/test_stability_basic.py` | `scope_stability`, `level_basic` |
| `tests/test_stability_metrics.py` | `scope_stability`, `level_basic` |
| `tests/test_stability_deep.py` | Historical `level_deep`, excluded from active execution |

An unclassified test path fails collection rather than silently
vanishing from the suite. Adding a new stability test file requires
an explicit scope/level assignment in `conftest.py`, and a reviewed
decision about its compute budget and reference inputs.

`--list` is an exact-level **display filter**. It does not have
the inclusive execution behavior of `--basic`, nor the special
all-unit rule of `--unit --smoke`:

```bash
python test.py --list
python test.py --list --integration --smoke
python test.py --list --numeric-stability --basic
```

The autouse test fixture resets Python's process-global `random`
and legacy NumPy RNG state before each test to protect isolation.
It does not make every separate local RNG or native library kernel
identical. Production component-seed derivation is protected by
[ADR 0045](adr/0045-component-seed-allocation.md).

## 6. Numerical references and CI

The numerical smoke file protects Python/NumPy PRNG signatures, the
experiment seed schedule, selected GT geometry and light solver fronts.
The basic file adds historical MOP/solver fronts and GT geometry tests
using their documented fixed inputs.

The conventional metric reference file adds frozen 2D/3D values for
GD, IGD, GD+, IGD+, EMD, and hypervolume
`raw`/`rel`/`abs`. In `abs` mode the GT is owned by the MOP.
The file checks exact HV and seeded approximate HV, where Monte
Carlo sampling error is treated differently from an exact numeric
comparison.

Different tests have different **explicit tolerances**. For example,
the existing seeded-front basic test checks exact shape followed by
`np.allclose(atol=1e-12, rtol=1e-8)`; fixed mathematical metric
values use `atol=1e-10, rtol=0`. There is no legitimate universal
"round to six decimals" shortcut for replacing them.

### 6.1 The two regular workflows

The repository has
[`unit-contract-tests.yml`](../.github/workflows/unit-contract-tests.yml)
and
[`stability-separation.yml`](../.github/workflows/stability-separation.yml).
They provide overlapping but complementary evidence.

The unit-contract workflow verifies unit contracts, GT integration,
numerical smoke and the full integration-basic suite. A separate
reference-numeric job checks all numerical basic fixtures.

The stability-separation workflow verifies the separation interface,
independent metric reference derivations, synthetic statistical v2
logic, full unit tests, a small performance-adapter integration and
numerical smoke. Its own reference-numeric job also checks the full
numerical basic suite.

Synthetic tests of the performance decision rule check whether the
statistics and CLI behave correctly; they do **not** provide historical
MOEA performance measurements. Both CI reference-numeric jobs verify
the effective CPU/BLAS profile before running their frozen fronts.

## 7. Preflight: establishing the numerical reference runtime

A *preflight* is a **preliminary environment check**, performed
before interpreting results or certifying a release.

The approved NSGA-III×DPF1 trajectory depends on both the NumPy
floating-point implementation selected during crossover and the
OpenBLAS kernel used during survival/normalization. Matching Python,
NumPy, SciPy and pymoo **version numbers alone** did not guarantee the
same trajectory across GitHub Actions hosts.

On the tested Linux/x86_64 reference CI images, the following
environment produced the historical front without altering the
algorithm, seeds, frozen values or regression tests:

```bash
export NPY_DISABLE_CPU_FEATURES=X86_V4
export OPENBLAS_CORETYPE=HASWELL
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# After installing the exact pinned numerical packages and threadpoolctl:
python scripts/check_numerical_reference_runtime.py
python test.py --numeric-stability --basic
```

The read-only preflight checks **actual effective execution**:
`numpy.power` must select `baseline(X86_V2)`, both relevant
OpenBLAS instances must select `Haswell`, and execution must be
single-threaded. An environment-variable string alone is not
sufficient evidence. The reference CI uses Python 3.14.4 and the
pinned numerical stack described in
[Environment verification](environment-verification.md).

These settings are limited to the **reference verification
environment**. The fact that they reproduced the historical tests
on Ubuntu 22.04 and 24.04 does not establish equivalence on every
processor and native BLAS build.

## 8. Scientific performance certification

The full predeclared study, measurement provenance, independent pilot,
sample-size power planning and binary statistical decision are in
[Performance certification](performance-stability.md) and
[ADR 0056](adr/0056-versioned-certification-and-performance-reference-lifecycle.md).

For a future candidate **distinct from the reference release**:

```bash
python test.py --certify-performance \
  --reference moeabench/resources/certifications/0.18.0/performance-reference.json \
  --output moeabench/resources/certifications/1.0.0/performance-vs-0.18.0.json
```

The reference must have an explicit review record and `APPROVED` status.
The candidate uses the original seeds and frozen objective-space
reference points; no new baseline is produced during verification.
An eight-pair pilot is used only for sample-size planning, and a
separate confirmatory sample supports simultaneous, familywise 1%
Student-t inference across six IGD+ cases. With no evidence sufficient
to certify all six cases, the release gate returns `NOT_CERTIFIED`
(nonzero exit), recording whether the cause is demonstrated regression,
insufficient evidence or insufficient statistical power.

The 0.1% per-case engineering tolerance and 95% planning power at a
true zero mean change are explicitly **bounded operational design
choices**, not mathematical claims of equality of distributions or
universal algorithm quality. Insufficient power under the 56-pair
confirmatory ceiling causes a negative certification outcome, never
a silent increase of the budget.

## 9. Numerical certification of a candidate version

`--certify-numeric` is a **formal developer operation**. It runs the
effective NumPy/OpenBLAS kernel preflight, reruns the entire unit,
integration and numerical-stability `basic` suite without inherited
`PYTEST_ADDOPTS` filters, and only after all checks pass updates the
current dependency manifest and registers a **compact immutable
versioned certificate** with software versions, runtime profile,
hashes of existing numerical reference files and execution evidence.

Nothing in this command generates a new solver-front baseline,
recalculates MOP GT, measures statistical performance or triggers
`--record-performance-baseline`.

For a clean checkout, integration example archives must be prepared
using the same deterministic setup as regular CI, since the full
`basic` suite requires them:

```bash
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

python scripts/check_numerical_reference_runtime.py
python test.py --certify-numeric
```

A failing preflight or `basic` batch prevents certification; no
historical expected values are replaced. The existing
[0.18.0 certification evidence](https://github.com/monacofj/moeabench/actions/runs/44351880965)
recorded 389 passing tests under the historical numeric command.
The canonical new numeric command has also been executed successfully
on the same original source revision:
[compact historical certificate run](https://github.com/monacofj/moeabench/actions/runs/44403553093).
The checked-in numerical archive has three fixture hashes and the
recorded execution environment; it **does not duplicate complete
numerical fronts** for each software version.

For a later package release, update the package source version,
test and certify its own numerical environment, and compare
performance separately with a previously approved reference. A
baseline is created only when independently authorized, not as
part of every numerical certification. Certification 0.18.0 did
not publish a Git tag or a 1.0.0 release.

## 10. Contribution and completion checklist

Before adding a test, identify the contract it protects and the
least expensive level that produces meaningful evidence. Add a unit
oracle for a small mathematical property, an integration test for
composition, a numerical-stability case for an independently
approved fixed output, and a statistical performance study only
when a reviewed reference and adequate repetitions exist.

New files under `tests/unit/` and `tests/integration/` acquire
the corresponding scope automatically; the explicit smoke
allowlists are maintained in `tests/conftest.py`. New
root-level numerical-stability files must be added to the
appropriate stability allowlist or collection fails. Inspect
classification with `python test.py --list` before running
the intended scope.

If a test fails, preserve the evidence. **Do not modify a
historical regression test, fixture or baseline without warning
the maintainer first and obtaining explicit authorization.**
A change contradicting an accepted ADR likewise requires prior
authorization and a documented architectural decision. Do not
relax tolerances, suppress cases or replace historical outputs
just to obtain a green CI result.

The completed verification work and its evidence are recorded
in [ADR 0050](adr/0050-certified-environment-verification-and-provenance.md),
the [numerical reference audit](audits/conventional-metric-reference-v1.md),
the closed issues [#97](https://github.com/monacofj/moeabench/issues/97),
[#98](https://github.com/monacofj/moeabench/issues/98) and
[#92](https://github.com/monacofj/moeabench/issues/92), and the
[environment guide](environment-verification.md).

**Remember the distinctions:** mathematics checked by unit tests;
workflows checked by integration; replay checked by numerical
stability; stochastic degradation checked by an externally
referenced statistical experiment; dependency versions registered
by formal certification. Completing one does not silently
complete or substitute for the others.


## 11. Versioned historical evidence

The **approved** inaugural 0.18.0 performance reference and the numerical
certificate are kept under
[`moeabench/resources/certifications/0.18.0/`](../moeabench/resources/certifications/0.18.0/).
The former contains individual performance observations and immutable
geometric inputs; the latter contains only the source/environment
snapshot and **hashes** of active frozen numerical regression files.

`mb.system.certification_history()` lists available historical
certificates; `mb.system.matching_certified_environments()` lists
exact dependency-version matches. A version match is **not**
evidence of identical numerical results on the user's MOP or equal
stochastic performance. Claims of agreement require an actual
comparison with recorded reference values under controlled
inputs. See [ADR 0056](adr/0056-versioned-certification-and-performance-reference-lifecycle.md).

The first baseline passed an independent automated audit of the original
SHA, all 384 observations and 18 repeat solver runs
([audit evidence](https://github.com/monacofj/moeabench/actions/runs/44409693329)).
Its approval records the technical review scope without representing
human review of every observation. **Future** candidate versions must
independently certify their performance against the frozen reference.
Re-certifying an already recorded package version with a different
source revision is refused: advance the package version first.
