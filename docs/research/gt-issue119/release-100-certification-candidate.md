<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
SPDX-License-Identifier: GPL-3.0-or-later
-->

# MoeaBench 1.0.0 numerical certification candidate — issue #119

**Status:** draft candidate, no released version and no performance verdict.

The existing `0.18.0` approved performance reference is immutable.
The changes merged into main since its original source revision
(`0d3e66804351e23b93fb483ff25e178e06ef6e12`) require
a distinct candidate version: `1.0.0` is staged here only,
on the release branch, pending approval of the complete release.

## Numeric certification (candidate-SHA bound)

The dedicated `release-100-numeric-certification.yml` job runs
`python test.py --certify-numeric` on the **specific checked-out
candidate commit**, using Python 3.14.4 and the approved native
NumPy/OpenBLAS kernel profile. It prepares integration sample archives
before certification. Only after all unit/integration/numerical basic
checks pass does the command write a new `1.0.0` environment manifest
and a *separate* compact numeric certificate in the runner workspace.

The workflow uploads those two outputs as an **unmerged CI artifact**
with the exact candidate SHA. It neither creates a baseline nor
overwrites the committed 0.18.0 reference. A generic CI pass is not
this formal certification; the complete certification command must
pass and produce matching SHA and fixture hashes.

GitHub squash-merging or editing the branch creates a **different
source revision**, invalidating a candidate-SHA certificate. The
release source revision must therefore be frozen and then certified
again, after the final source commit is chosen, before certification
outputs can be accepted as release evidence. Embedding a certificate
that includes its own source commit hash in that very same Git
commit is impossible without changing the hash; a versioned CI
artifact or approved external immutable evidence store is therefore
the initial evidence location.

## Performance certification is independent

Performance certification is **not triggered automatically** by
numeric certification. It must consume the approved, immutable
0.18.0 baseline and its original frozen objective references,
the exact candidate numeric certificate, and the same approved
Python/BLAS environment. It uses the fixed six-case ADR 0056
protocol: 64 historical seeds/case, eight independent pilot
pairs excluded from confirmation, up to 56 confirmatory pairs,
0.1% materiality, 1% familywise alpha and binary
`CERTIFIED`/`NOT_CERTIFIED`. A lack of statistical power is
`NOT_CERTIFIED`, never a license to refresh the baseline.

The finite-GT and two-seed pilot studies of issue #119 support
correctness and reference sensitivity but are **not** substitutes
for the ADR 0056 performance protocol.

## Required release decision

The release-candidate workflow runs on a qualifying push to
`main` after the PR is integrated, so its new numeric certificate
identifies the **actual main merge commit**. The independent
performance workflow has no numeric-workflow dependency or
automatic completion trigger. It is opt-in through an exact,
maintainer-authored issue #119 comment:

```text
/certify-performance-119 <full 40-character main source SHA>
```

The workflow refuses commands for any other checked-out revision,
looks up a separate completed numeric-certification workflow for the
same source SHA, downloads its certificate/manifest, and requires
the full source, environment, CPU-profile and original regression
fixture digest checks. It then executes the existing six-case ADR
0056 certification CLI against the frozen historical 0.18.0
performance artifact. A failed or underpowered study uploads its
diagnostics but cannot publish a `CERTIFIED` verdict.

The project owner must review the exact final candidate SHA,
version name, experimental environment and certification outputs
before a public 1.0.0 release or registration of the first
1.0.0 performance certificate. This draft authorizes no automatic
release, tagging or baseline update.
