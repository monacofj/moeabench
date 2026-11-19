<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
SPDX-License-Identifier: GPL-3.0-or-later
-->

# Contributor and coding-agent instructions

These instructions apply to every change in this repository, including
changes proposed by automated coding assistants. They do not override
accepted architecture decisions or the maintainer's explicit instructions.

## Read before planning or editing

1. Read [the ADR index](docs/adr/README.md). Find the **current normative
   decisions** affecting the requested component. Follow supersession links,
   partial supersessions and dated addenda **through to the end** of the
   applicable ADRs. An older proposal is not authoritative merely because
   it is easier to find.
2. Inspect the *current main-branch implementation*, public API,
   relevant tests and existing issues/PRs. Record what is already
   implemented, what is decided but pending, and what is genuinely open.
3. For scientific work, separate mathematical correctness, numerical
   reproducibility, empirical evidence and performance certification.
   Historical frozen data are evidence of previous behavior, not
   automatically correct scientific truth.
4. Before suggesting a new architecture, justify why the adopted
   architecture and existing interfaces cannot solve the problem.
   Avoid reopening accepted decisions by default.

## Explicit maintainer authorization barriers

- **If a proposed change contradicts an accepted ADR, warn the maintainer,
  identify the specific conflicting clause and wait for explicit approval
  before implementing the incompatible change.** Record the supersession
  in a new accepted ADR when approved; preserve historical ADR text.
- **Never modify, delete, relax, regenerate or replace an existing
  regression test, frozen fixture, numeric/performance baseline or
  certification reference without first warning the maintainer** and
  obtaining authorization for the affected change.
- New independent tests and non-mutating diagnostics are permitted if
  they do not redefine existing acceptance contracts.
- Do not automatically regenerate baselines or declare statistical
  performance stable because a single deterministic run passes.

## Change procedure

Before opening a PR, state:

- Relevant ADRs, current status, and exact normative provisions;
- existing public contracts and preserved invariants;
- any contradiction and the approval that resolves it, or mark the
  incompatible work **blocked**, leaving accepted behavior intact;
- tests added versus existing regression tests touched, with reasons
  and prior authorization;
- independent validation evidence and reproducible commands;
- whether data, calibration references, persistence or release
  certification are affected.

Keep investigations, planned decisions and accepted implementation
distinct in issue/PR descriptions. Never treat a *Proposed* ADR as an
effective change of policy.

## Relevant component entry points

This map is a navigation aid, **not** a substitute for reading the ADRs:

- MOPs and analytical PS/PF: ADR 0006; ADR 0016 (supersedes 0015);
  ADR 0017; ADR 0018; current GT investigation #119 and DTLZ5/6 #65.
- GT calibration and scientific provenance: ADR 0017/0018 and current
  `moeabench/mops/calibration.py`; proposed change #119 is not adopted.
- Experiment save/load: ADR 0011, ADR 0032 and current
  `moeabench/core/save.py`; inspect all persistence modes.
- Views: ADR 0014, ADR 0040, applicable API decisions and
  `moeabench/view/topo.py`; do not equate plot sampling with scientific
  calibration.
- HV: ADR 0019, 0035, 0047 and applicable normalization/engine ADRs;
  research #89 does not change the established contract.
- Numerical/performance certification: ADR 0050 and **ADR 0056 including
  its final dated addenda**, and `docs/test.md`.
- FAIR historical research: proposed ADRs 0051–0055, archived or
  deferred; do not reactivate FAIR as a core GT dependency.

When in doubt, state what was verified, and identify unresolved
decisions rather than silently choosing a new default.
