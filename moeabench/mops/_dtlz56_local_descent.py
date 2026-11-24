# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent first-order, zero-gap local dominance witnesses for DTLZ5/6.

Let u=(x_1,...,x_{M-1}) and parameterize feasible objectives F(g,u).
At a candidate (g0,u0), choose a feasible one-sided path
u(g)=u0+(g-g0)*v. If dF_i(g,u(g))/dg at g0 is strictly positive
for every i, reducing g by a sufficiently small *positive* amount
strictly decreases all objectives (continuity of derivatives).
This is an exact SUFFICIENT theorem for non-Pareto-optimality, not an
equivalence or a criterion for positive membership. An LP searches
for such a directional v, then an independent numerical witness is
reconstructed. All numerical results are provisional under ADR 0059.

No production MOP evaluator, PS/PF or scalarizer is called.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import linprog


@dataclass(frozen=True)
class LocalDescentAudit:
    status: str
    g_candidate: float
    derivative_margin: float | None
    position_direction: tuple[float, ...] | None
    witness_g: float | None
    witness_objectives: tuple[float, ...] | None
    witness_decision: tuple[float, ...] | None
    min_objective_improvement: float | None
    basis: str = (
        "Independent analytic one-sided derivative sufficiency for "
        "strict local Pareto dominance; LP/numerical witness require "
        "source-bound environment and output verification; no "
        "negative global Pareto certificate"
    )

    def to_dict(self):
        return {
            "status": self.status,
            "g_candidate": self.g_candidate,
            "derivative_margin": self.derivative_margin,
            "position_direction": (
                None if self.position_direction is None
                else list(self.position_direction)),
            "witness_g": self.witness_g,
            "witness_objectives": (
                None if self.witness_objectives is None
                else list(self.witness_objectives)),
            "witness_decision": (
                None if self.witness_decision is None
                else list(self.witness_decision)),
            "min_objective_improvement": self.min_objective_improvement,
            "basis": self.basis,
        }


def _objective_and_derivative(u, g):
    """Published positive hyperspherical F and analytic dF/dg at u'(g)=v.

    Return F, b, A where dF/dg = b + A @ v. Product differentiation
    (rather than logarithmic differentiation) handles zero coordinates.
    """
    m = len(u)+1
    theta = np.empty(m-1)
    theta[0] = np.pi*u[0]/2
    theta[1:] = np.pi*(1+2*g*u[1:])/(4*(1+g))
    h = np.empty(m)
    jac = np.zeros((m,m-1))
    for j in range(m):
        factors = np.ones(m-1)
        if j < m-1:
            factors[:m-j-1] = np.cos(theta[:m-j-1])
        if j > 0:
            factors[m-j-1] = np.sin(theta[m-j-1])
        h[j] = np.prod(factors)
        for k in range(m-1):
            if k < m-j-1:
                d = -np.sin(theta[k])
            elif j > 0 and k == m-j-1:
                d = np.cos(theta[k])
            else:
                continue
            jac[j,k] = d*np.prod(np.delete(factors,k))
    fixed = np.zeros(m-1)
    fixed[1:] = np.pi*(2*u[1:]-1)/(4*(1+g)**2)
    scaling = np.empty(m-1)
    scaling[0] = np.pi/2
    scaling[1:] = np.pi*g/(2*(1+g))
    return (1+g)*h, h+(1+g)*jac@fixed, (1+g)*jac*scaling[None,:]


def _reconstruct_decision(u, g, *, problem, k):
    m = len(u)+1
    x = np.empty(m+k-1, dtype=float)
    x[:m-1] = u
    if problem == "DTLZ5":
        x[m-1:] = .5+np.sqrt(g/k)
    else:
        x[m-1:] = (g/k)**10
    return x


def audit_local_dtlz56_descent(
    decision, *, problem, m, numerical_tolerance=1e-10,
    max_halvings=30,
):
    """Find a finite witness guaranteed by a strict first-order condition.

    A result of 'witness' implies objective-space strict improvement
    in the independently reconstructed numerical model. A result of
    'unverified' says nothing about global Pareto optimality.
    """
    x = np.asarray(decision, dtype=float)
    if problem not in ("DTLZ5", "DTLZ6") or m < 2:
        raise ValueError("Original DTLZ5/6 with M>=2 required")
    if x.ndim != 1 or x.size < m or not np.isfinite(x).all():
        raise ValueError("Expected one finite decision vector, N>=M")
    if np.any(x < 0) or np.any(x > 1):
        raise ValueError("Decision values must lie in [0,1]")
    if not np.isfinite(numerical_tolerance) or numerical_tolerance <= 0:
        raise ValueError("numerical_tolerance must be positive and finite")
    if not isinstance(max_halvings,int) or max_halvings < 1:
        raise ValueError("max_halvings must be >=1")
    k = x.size-m+1
    tail = x[m-1:]
    g = (float(np.sum((tail-.5)**2)) if problem=="DTLZ5"
         else float(np.sum(tail**.1)))
    if g <= numerical_tolerance:
        return LocalDescentAudit(
            "unverified",g,None,None,None,None,None,None)
    u = x[:m-1]
    f, b, a = _objective_and_derivative(u,g)
    # With decreasing g: x_i(g)=u_i+(g-g0)*v_i. At x_i=0
    # require v_i<=0; at x_i=1 require v_i>=0.
    bounds = []
    for value in u:
        if value <= numerical_tolerance:
            bounds.append((None,0))
        elif value >= 1-numerical_tolerance:
            bounds.append((0,None))
        else:
            bounds.append((None,None))
    bounds.append((None,None))  # maximize the common derivative floor t
    c = np.r_[np.zeros(m-1), -1.]
    result = linprog(
        c, A_ub=np.c_[-a, np.ones(m)], b_ub=b,
        bounds=bounds, method="highs")
    if not result.success or result.x[-1] <= 32*numerical_tolerance:
        return LocalDescentAudit(
            "unverified",g,
            float(result.x[-1]) if result.success else None,
            tuple(map(float,result.x[:-1])) if result.success else None,
            None,None,None,None)
    v = result.x[:-1]
    derivative = b+a@v
    if np.min(derivative) <= 32*numerical_tolerance:
        return LocalDescentAudit(
            "unverified",g,float(np.min(derivative)),
            tuple(map(float,v)),None,None,None,None)
    # A strictly positive derivative gives an exact local existence
    # theorem. Provide an independent *finite* numeric instance, never
    # infer an unsafe witness merely from LP optimality.
    initial_step = min(g/4, .01)
    for shift in range(max_halvings):
        eps = initial_step*2.0**(-shift)
        gp = g-eps
        up = u-eps*v
        if gp < 0 or np.any(up < 0) or np.any(up > 1):
            continue
        candidate,_,_ = _objective_and_derivative(up,gp)
        improvements = f-candidate
        if float(np.min(improvements)) > numerical_tolerance:
            witness = _reconstruct_decision(up,gp,problem=problem,k=k)
            return LocalDescentAudit(
                "witness",g,float(np.min(derivative)),
                tuple(map(float,v)),float(gp),
                tuple(map(float,candidate)),
                tuple(map(float,witness)),
                float(np.min(improvements)))
    return LocalDescentAudit(
        "unverified",g,float(np.min(derivative)),
        tuple(map(float,v)),None,None,None,None)
