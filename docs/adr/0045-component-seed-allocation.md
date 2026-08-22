<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# ADR 0045: Deterministic Component Seed Allocation

## Status

Accepted

## Context

An optimization run may contain more than one stochastic component. In the
NSGA-III wrapper, pymoo's evolutionary process and the energy-based reference-
direction generator previously received the same numeric seed. Although this
was reproducible, it coupled conceptually independent random streams and made
controlled experiments difficult: changing the run seed necessarily changed
the reference directions, while fixing the directions required bypassing the
wrapper's ordinary configuration model.

Adding a universal collection of component-specific parameters to the public
MOEA API would expose algorithm internals and make the common interface grow
with every new stochastic component. Allocating seeds sequentially from a
shared RNG would also make results depend on component initialization order.

## Decision

- The common API retains one universal run `seed`.
- Internal stochastic components receive deterministic seeds derived from the
  run seed and a stable namespaced component identifier.
- Component derivation is stateless and order-independent. Version 1 hashes
  `moeabench-seed-v1\0{run_seed}\0{component}` with SHA-256 and interprets the
  first four digest bytes as an unsigned big-endian integer.
- If the resulting 32-bit value equals the run seed, it advances by one modulo
  `2**32`, ensuring a distinct numeric seed in the applicable range.
- The canonical NSGA-III component identifier is
  `nsga3.reference_directions`.
- U-NSGA-III follows the same allocation policy under the distinct canonical
  identifier `unsga3.reference_directions`.
- `NSGA3(ref_dirs_seed=None)` and omission derive the reference-direction seed
  automatically for each run. `NSGA3(ref_dirs_seed=<int>)` uses that fixed
  32-bit seed for all runs.
- Wrapper-specific keywords are consumed from a copy of the algorithm kwargs;
  they are neither forwarded to pymoo nor removed from persistent wrapper
  configuration.
- `run.seed` remains the evolutionary seed. Effective internal seeds are copied
  into `run.component_seeds`, reports, serialized experiment state, and the
  per-run entries of `metadata.json`.
- Algorithms without registered stochastic components retain an empty mapping
  and do not add empty component sections to reports.

## Consequences

- Equal run configurations reproduce both the evolutionary seed and generated
  reference directions without consuming or advancing one another's RNG.
- Changing `ref_dirs_seed` does not change the seed passed to pymoo's
  evolutionary process, but different directions can still change the search
  trajectory and final result.
- Researchers can hold NSGA-III reference directions constant across repeated
  evolutionary seeds without adding an NSGA-III parameter to `Experiment` or
  `BaseMoeaWrapper`.
- Namespaced metadata can accommodate future algorithm components without
  collisions or a universal component-seed API.
- The derivation string and version prefix are reproducibility contracts.
  Changing them requires a new derivation version and an explicit migration
  decision rather than silently changing historical results.
- Existing archives remain loadable because `Run.component_seeds` treats a
  missing legacy field as an empty mapping; the persistence schema gains only
  additive metadata.

## Alternatives considered

- **Reuse the run seed:** reproducible, but preserves unnecessary stream
  coupling and prevents independent experimental control.
- **Draw component seeds from a master RNG:** produces separate values, but
  makes allocation dependent on call order and future component additions.
- **Expose universal `component_seeds`:** flexible, but leaks algorithm-specific
  structure into the common API.
- **Keep reference directions fixed by default:** isolates evolutionary noise,
  but no longer represents a new stochastic execution of NSGA-III as a whole.
