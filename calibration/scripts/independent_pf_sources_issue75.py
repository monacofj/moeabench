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
