# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np

class GEN_mc_hypervolume:
    """
    Monte Carlo approximation of Hypervolume using vectorized NumPy operations.
    Useful for many-objective optimization (M > 6).
    """
    def __init__(self, hist_F, M, approx_ideal, approx_nadir, n_samples=100000, **kwargs):
        self.hist_F = hist_F
        self.M = M
        self.ideal = approx_ideal
        self.nadir = approx_nadir
        self.n_samples = n_samples

    def evaluate(self):
        # Fixed normalized reference point, consistent with GEN_hypervolume.
        ref_point = np.full(self.M, 1.1)

        results = []
        for F in self.hist_F:
            if len(F) == 0:
                results.append(0.0)
                continue
            
            # 2. Normalize population to [0, 1] based on ideal/nadir
            # (already done by the caller in the evaluation pipeline usually, 
            # but we ensure it here if passed as raw floats)
            # Actually, the pipeline passes normalized F if it's following the GEN_ pattern.
            F_norm = (F - self.ideal) / (self.nadir - self.ideal + 1e-10)

            # Preserve fixed-reference semantics for out-of-bounds data. Better-than-
            # ideal points may extend dominated volume below zero; worse-than-reference
            # points contribute nothing. Bounds are never expanded or recomputed.
            lower = np.minimum(0.0, np.min(F_norm, axis=0))
            box_volume = np.prod(ref_point - lower)
            samples = np.random.uniform(lower, ref_point, (self.n_samples, self.M))
            
            # 4. Check dominance: a sample is "dominated" if there exists a point p in F_norm 
            # such that p_i <= sample_i for all i.
            
            # Vectorized check:
            # We want to know for each sample if it is dominated by ANY point in F_norm.
            # Using broadcasting: (n_samples, 1, M) >= (1, n_pop, M)
            # This can be memory intensive for huge samples/pop.
            # Efficient implementation:
            is_dominated = np.zeros(self.n_samples, dtype=bool)
            
            # For each point in the Pareto front, mark the samples it dominates
            for p in F_norm:
                # A sample is dominated by p if sample >= p in all dimensions
                dominated_by_p = np.all(samples >= p, axis=1)
                is_dominated |= dominated_by_p
                # Early exit if all samples are already dominated
                if np.all(is_dominated):
                    break
            
            # 5. Proportion of dominated samples
            hv_approx = (np.sum(is_dominated) / self.n_samples) * box_volume
            results.append(float(hv_approx))
            
        return np.array(results)
