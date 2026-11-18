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
versions can still yield diverging optimization trajectories. Earlier
NSGA-III/DPF1 experiments detected `1e-15`-scale drift in reference directions,
and an isolated rounding experiment produced matching trajectories. The
production wrapper still does **not** round directions. Follow-up controlled
tests identified differences in NumPy SBX `power` dispatch and in OpenBLAS
selection/normalization kernels as causes of the **observed regression**;
the controlled reference profile described below reproduced the original
front. This result does not establish bitwise portability of all problems.
See [ADR 0050](adr/0050-certified-environment-verification-and-provenance.md).

Matplotlib remains outside the certified numerical dependency manifest;
functional plotting compatibility is tested in the integration suite.

## Developer certification

From the developer reference environment, run `python test.py --basic` (unit, integration and numerical stability test suites). Then run `python test.py --certify-numeric`, which reruns the complete basic suite and updates `moeabench/resources/certified_environment.json` **only on success**. Review and commit this manifest with the appropriate release; it does not overwrite numerical baselines. The legacy `--deep` statistical test has no usable baseline and is not accepted as release evidence. For a separate, explicit performance assessment, use `python test.py --certify-performance --reference PATH --output CERT.json` with an approved historical reference; see [Performance stability](performance-stability.md). A numerical failure never triggers this statistical assessment automatically.

Certified versions are not inferred from package dependency lower bounds or a random user's installation. If no developer-certified manifest has been committed, the status is UNCONFIGURED and certification is pending.

The canonical front fixtures shipped in the installed wheel mirror the smoke reference fixtures used in the development suite. Developers must review reference changes, not refresh failing fixtures automatically.


## Reproducing the numerical reference runtime (issue #92)

The numerical **basic** suite uses frozen reference fronts. Its NSGA-III/DPF1
case is sensitive to low-level `numpy.power` dispatch in pymoo's SBX operator
and to the OpenBLAS core used during NSGA-III survival/normalization.
Python, NumPy, SciPy and pymoo version identity alone did not reproduce the
frozen result across different GitHub Actions hosts.

The [issue #92 controlled CI experiment](https://github.com/monacofj/moeabench/actions/runs/44340186293)
reproduced the exact historical three-point NSGA-III/DPF1 front, **without any
regression or solver code change**, under both Ubuntu 22.04 and 24.04 when
the numerical reference jobs used this **Linux/x86_64 reference profile**:

```bash
export NPY_DISABLE_CPU_FEATURES=X86_V4
export OPENBLAS_CORETYPE=HASWELL
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# After installing the dependencies pinned by the reference CI and threadpoolctl:
python scripts/check_numerical_reference_runtime.py
python test.py --numeric-stability --basic
```

The preflight checks the **actual** NumPy float64 power dispatch
(`baseline(X86_V2)`), OpenBLAS kernel (`Haswell`), and thread count, not
merely the requested environment variable strings. Its failure means the
host cannot claim equivalence with the tested reference profile.

The pinned numerical reference CI uses this profile; routine unit and smoke
tests remain free to run in a developer's ordinary environment. The
regular [unit-contract workflow](https://github.com/monacofj/moeabench/actions/runs/44341867393)
and [stability-separation workflow](https://github.com/monacofj/moeabench/actions/runs/44341867318)
each passed **41/41 basic numerical regression cases** after applying the
profile, as well as their unit/contract/integration jobs. This is a
**reproducibility profile for reference verification**, not an instruction to
change all user computations or proof of bitwise portability across arbitrary
hardware, BLAS builds and operating systems. A user running outside this
profile may still run experiments, but matching package versions alone
must not be interpreted as numerical equivalence (ADR 0050).

The existing **numerical regression fronts remain unchanged**. The
developer-certified environment manifest has now been updated for the
intermediate **0.18.0 candidate** by executing both the complete
`--basic` suite and the actual `--certify` rerun successfully in the
designated reference profile: **389/389** tests passed in each run.
See [certification evidence #106](https://github.com/monacofj/moeabench/actions/runs/44351880965).
The previous 0.16.1 manifest remains recoverable in Git history.

The original certification attempt failed on **missing integration
example archives**, correctly blocking a manifest update. The ordinary
unit-contract CI generates `example_full_exp1.zip` and
`example_full_exp2.zip` before integration; the successful
certification reproduced that prerequisite exactly. See the
[complete testing manual](test.md) for the necessary archive-generation
commands, scopes, and release safeguards.

The current `--certify-numeric` command replays the full basic suite and checks the effective native profile; the original historical `--certify` command
does not itself inspect CPU/BLAS dispatch; the explicit preflight is
required for the designated profile. For the future 1.0.0 version,
the software version must be updated and the complete certification
rerun rather than editing the version label in the manifest. Neither
the 0.18.0 certification nor this version record constitutes an
independently reviewed statistical performance baseline.
