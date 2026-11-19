<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
SPDX-License-Identifier: GPL-3.0-or-later
-->

# Global Pareto membership and full attainable support of the published DPF5 — issue #119

**Scope:** exact mathematical characterization of the **published reconstructed DPF5**, not certification of the current implementation nor of a finite sample. The derivation assumes minimization of all M objectives; standard box decision constraints 0<=x_i<=1; M>D>=2; N>=M; and the paper's distance function `g=(x_M-x_1)^2 + sum_{i=M+1}^N (x_i-0.5)^2`. It also assumes published beta/P for x1<1/3 and gamma/D for x1>=1/3, with the hyperspherical coordinates implemented independently by `scripts/audit/gt_dpf5_general_d_issue119.py`.

## Analytic characterization

Let `H_q` be the standard nonnegative unit-hyperspherical map in q objectives, with q−1 angles. For θ_i=(π/2)x_i, the published P branch uses the complete M-dimensional hyperspherical image `u_P=H_M(theta_1,...,theta_{M-1})`. For the degenerate D branch, let `h=H_D(theta_1,...,theta_{D-1})`; the first coordinate `h_1` is replicated M−D+1 times with weight `1/sqrt(M−D+1)`, followed by `h_2,...,h_D`. Hence both vectors `u(x)` have nonnegative components and unit Euclidean norm. The published objective equations are exactly

```text
F(x) = (1 + g(x)) u(x),   g(x) >= 0,   ||u(x)||_2 = 1.
```

**Proof — all g=0 points are globally nondominated.** If y has g(y)=0, ||F(y)||=1. Any feasible x with F(x) componentwise <= F(y), with one strict, must satisfy ||F(x)||<||F(y)||=1 because every component is nonnegative. But the published equations guarantee ||F(x)||=1+g(x)>=1. Contradiction. This includes any cross-branch candidate; thus there is **no cross-branch dominance among points of either P/D at g=0**, even when objective vectors coincide on region intersections.

**Proof — every g>0 point is dominated.** Given any feasible x, let y share all angular coordinates x_1,...,x_{M-1} and all other non-distance coordinates, but set y_M=x_1 and y_i=0.5 for i>M. Then y is box-feasible (N>=M), g(y)=0, preserves x1 and the published branch choice, and u(y)=u(x). Hence F(x)=(1+g(x))F(y), and g(x)>0 implies F(y)<=F(x) componentwise with at least one strict inequality (u has norm one, so some component is positive). Thus x cannot be Pareto-optimal when g(x)>0.

**Conclusion:** the exact Pareto **decision set** is (\{x:g(x)=0\}\), with free x_1,...,x_{M-1}, x_M=x_1, all x_i=0.5 for i>M; its objective image under the branch map is the exact **whole global PF**. The P portion is the x1<1/3 spherical region, and the D portion is the x1>=1/3 projected/degenerate spherical region. P's strict boundary is generally not attained from P at x1=1/3; D's boundary is included. The objective-space PF may have coincident points or differing geometric strata. This is a **set-theoretic completeness proof for the published equations**, not evidence of uniform sampling, finite-cardinality quality, or regularity.

## What this does and does not settle

- **Settles:** For the mathematical DPF5 as published, global Pareto optimality/membership is equivalent to g=0; all attainable points of the global PF are exactly parameterized by that analytic PS, including both strict and nonstrict branches. Geometric completeness as a parameterization is analytically established.
- **Does not settle:** Exact equality of the production DPF5 evaluator and published equations (already falsified for several M,D); validity/coverage/cardinality of the current sampled `ps()/pf()`; selection of representative finite points on mixed-dimensional strata; exact branch measure, regularity, discretization tolerance or calibration budget; numerical evaluation precision at extreme angles; release migration or current archived GT validity.
- **Boundary:** for exactly x1=1/3, the published D branch applies, and the proof uses that. A finite grid can sample equality exactly, unlike independent continuous random draws.
- **Tests:** `scripts/audit/gt_dpf5_global_membership_issue119.py` uses feasible randomized decisions including boundary/nextafter controls and constructs a matching same-angle g=0 dominating witness; `tests/unit/test_gt_dpf5_global_membership_issue119.py` adds new, separate numerical witness assertions. The proof above provides the mathematical argument; a finite randomized test by itself would not establish the theorem.
- **ADRs:** no production code, accepted ADR, existing regression, frozen GT or calibrated sidecar changes are authorized. ADR 0006/0017/0018/0056/0057 protections remain in force; the sampling and migration decisions remain separate, as described in `docs/gt-dpf-correction-migration-proposal-issue119.md`.

**Next bounded step:** derive or certify the *finite reference sampler* to cover both P/D supports, including branch boundary and extremes, without treating a discrete sample's local nondominance as an independent proof. A rigorously complete published PF parameterization means that a finite reference can be assessed against it; it does not establish finite GT density automatically.
