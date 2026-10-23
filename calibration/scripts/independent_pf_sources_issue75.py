#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Independent-source FAIR ideal_res challenge (issue #75).

Independent constructions DO NOT call MOP.pf()/ps() or any MoeaBench Pareto
sampling helper:
* DTLZ1 M=3: Dirichlet(1,1,1) uniform simplex surface, sum objectives = .5.
* DTLZ2 M=3: direct area-uniform positive unit-sphere octant: z~U[0,1],
  azimuth~U[0,pi/2], x/y = sqrt(1-z^2) cos/sin(azimuth).
* DPF5 M=3: random feasible Pareto-set decision inputs, evaluated through
  the existing MOP objective function. This is an *exploratory* independent
  decision-space sampler; its induced objective-surface density is UNKNOWN,
  and no claim of uniform-area sampling is made.

Frozen GT, ideal/nadir and MOP surface identities are validated. The returned
candidate objectives are NOT a replacement certified GT or Q baseline.
"""
import numpy as np
from scipy.spatial import cKDTree


def sample_independent_pf(name, n_points, seed, *, mop=None):
    """Return a fresh M=3 physical-objective candidate set and its law."""
    if not isinstance(n_points, int) or n_points < 2:
        raise ValueError("Independent sampler needs at least two points")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("Independent sampler needs a nonnegative private seed")
    rng = np.random.default_rng(seed)
    if name == "DTLZ1":
        # Independent direct objective geometry, not the library's
        # random Pareto-set x1/x2 map (which is not uniform on simplex).
        F = 0.5 * rng.dirichlet(np.ones(3), size=n_points)
        law = "uniform_simplex_area_dirichlet_111"
    elif name == "DTLZ2":
        # In spherical coordinates the sphere's surface-area Jacobian
        # makes z uniform for the 3D unit sphere; using uniform elevation
        # would give a biased concentration near the pole.
        z = rng.random(n_points)
        phi = (np.pi/2) * rng.random(n_points)
        radial = np.sqrt(np.maximum(0.0, 1.0-z*z))
        F = np.column_stack((radial*np.cos(phi), radial*np.sin(phi), z))
        law = "uniform_positive_sphere_area_z_azimuth"
    elif name == "DPF5":
        if mop is None:
            raise ValueError("DPF5 independent objective evaluation requires a MOP")
        if int(mop.M) != 3 or int(mop.N) < 3:
            raise ValueError("DPF5 independent decision sampler requires M=3,N>=3")
        X = np.full((n_points, int(mop.N)), 0.5, dtype=float)
        X[:, :2] = rng.random((n_points,2))
        X[:, 2] = X[:, 0]  # g=0 on the known Pareto-set construction.
        F = np.asarray(mop.evaluation(X)["F"], dtype=float)
        law = "exploratory_uniform_pareto_set_decisions_not_surface_area"
    else:
        raise ValueError(f"No externally constructed M=3 sampling law for {name}")
    if F.shape != (n_points,3) or not np.all(np.isfinite(F)):
        raise ValueError("Invalid independent sample objectives")
    return F, law



def sample_dtlz2_uniform_normalized_surface(n_points, seed, *, ideal, nadir):
    """Independent DTLZ2 PF sampling uniform by area *after* normalization.

    DTLZ2's physical Pareto front is the positive unit sphere. Let
    T(f)=(f-ideal)/(nadir-ideal), with the frozen package's affine bounds.
    Uniform physical-sphere area is NOT uniform on T(PF) when the coordinate
    spans differ. For an invertible diagonal map A=diag(1/span), its induced
    (M-1)-surface-area Jacobian is

        J_A(n) = abs(det(A)) * ||A^(-T) n||,

    where n is the unit sphere normal (= the raw sphere point). The constant
    determinant cancels during rejection sampling. A direction proposed
    uniformly on the physical positive sphere is accepted with probability
    ||span*n|| / max(span).

    This is a *research-only* sampler for a candidate normative measure,
    not a certified FAIR baseline or a replacement for DTLZ2's standard PF
    generator. Supports M >= 2 (also tests dimensional generalization).
    """
    if not isinstance(n_points, int) or n_points < 2:
        raise ValueError("At least two independent surface points required")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("Require a nonnegative private seed")
    ideal = np.asarray(ideal, dtype=float)
    nadir = np.asarray(nadir, dtype=float)
    if (ideal.ndim != 1 or nadir.shape != ideal.shape
            or ideal.size < 2
            or not np.all(np.isfinite(ideal))
            or not np.all(np.isfinite(nadir))):
        raise ValueError("Require finite matching ideal/nadir of dimension >= 2")
    span = nadir - ideal
    if np.any(span <= 0) or not np.all(np.isfinite(span)):
        raise ValueError("Normalized objective spans must be finite and positive")

    rng = np.random.default_rng(seed)
    accepted = []
    remaining = n_points
    # A private RNG ensures reproducibility and no process-global RNG mutation.
    while remaining:
        batch = max(256, min(65536, 4 * remaining))
        direction = np.abs(rng.standard_normal((batch, ideal.size)))
        norms = np.linalg.norm(direction, axis=1)
        direction /= norms[:, None]
        weighted_area = np.linalg.norm(direction * span, axis=1)
        take = rng.random(batch) * np.max(span) <= weighted_area
        new = direction[take][:remaining]
        if len(new):
            accepted.append(new)
            remaining -= len(new)

    physical = np.concatenate(accepted, axis=0)
    normalized = (physical - ideal) / span
    if not np.all(np.isfinite(normalized)):
        raise ValueError("Normalized surface candidate contains nonfinite points")
    return normalized, "uniform_normalized_dtlz2_surface_area_rejection"


def surface_identity(name, raw_F):
    """Independent, known exact geometry for DTLZ1/2; no such claim for DPF5."""
    F = np.asarray(raw_F,dtype=float)
    if F.ndim != 2 or F.shape[1] != 3 or not np.all(np.isfinite(F)):
        raise ValueError("Expected finite, 3-objective independent surface")
    if np.any(F < -1e-12):
        raise ValueError("Analytical objectives must be nonnegative")
    if name == "DTLZ1":
        deviation = np.abs(np.sum(F,axis=1)-0.5)
        accepted = np.max(deviation)<5e-12
        law = "simplex_sum_0.5"
    elif name == "DTLZ2":
        deviation = np.abs(np.linalg.norm(F,axis=1)-1.0)
        accepted = np.max(deviation)<5e-12
        law = "sphere_radius_1"
    elif name == "DPF5":
        return {"test":"not_analytically_verified","maximum_deviation":None,
                "passed":None}
    else:
        raise ValueError("Unsupported surface")
    if not accepted:
        raise ValueError(f"{name}: independent analytical surface identity failed")
    return {"test":law,"maximum_deviation":float(np.max(deviation)),"passed":True}


def normalized_independent_pf(name, *, n_points, seed, ideal, nadir, gt,
                              mop=None):
    """Validate physical geometry, normalize via exact frozen package and screen.

    A near-GT nonoverlap check is essential: FPS resampling from the same GT
    would give a fictitious zero analytical residue. Geometry screening is
    evidence, not an equivalence certificate of any two sampling measures.
    """
    F, law = sample_independent_pf(name,n_points,seed,mop=mop)
    surface = surface_identity(name,F)
    ideal=np.asarray(ideal,dtype=float)
    nadir=np.asarray(nadir,dtype=float)
    GT=np.asarray(gt,dtype=float)
    span=nadir-ideal
    if ideal.shape!=(3,) or nadir.shape!=(3,) or np.any(span<=0):
        raise ValueError("Incompatible frozen ideal/nadir")
    if GT.ndim!=2 or GT.shape[1]!=3 or len(GT)<3 or not np.all(np.isfinite(GT)):
        raise ValueError("Incompatible frozen GT")
    normalized=(F-ideal)/span
    distances=cKDTree(GT).query(normalized,k=1)[0]
    threshold=max(1e-12,1e-10*float(np.max(np.ptp(GT,axis=0))))
    # Independent generators can rarely coincide with an exact GT axis or
    # vertex. Discard those points rather than laundering them as 'fresh'.
    fresh=normalized[distances>threshold]
    if len(fresh)<n_points*0.95:
        raise ValueError(f"{name}: too few independent candidates after GT exclusion")
    spacing=float(np.median(cKDTree(GT).query(GT,k=2)[0][:,1]))
    if spacing<=0:
        raise ValueError("Invalid archived GT spacing")
    delta=cKDTree(GT).query(fresh,k=1)[0]
    mismatch={"median_GT_spacing":float(np.median(delta)/spacing),
              "p95_GT_spacing":float(np.percentile(delta,95)/spacing),
              "fresh_count":len(fresh),"requested_count":n_points}
    return fresh, {"law":law,"surface":surface,
                   "mismatch":mismatch,"certified":False}
