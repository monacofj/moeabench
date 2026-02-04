# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from scipy.stats import linregress, wasserstein_distance
from functools import cached_property
from typing import Dict, Any
from .base import StatsResult, SimpleStatsValue

class CasteSummary:
    """
    Helper object to access caste statistics using intuitive methods.
    """
    def __init__(self, data: Dict[int, Dict]):
        self._data = data
        
    def n(self, rank: int) -> int:
        """Returns the population count for the given rank."""
        return self._data.get(rank, {}).get('n', 0)
        
    def q(self, rank: int, level: float = 50) -> float:
        """
        Returns the quality (merit) level for the given rank.
        level=50 (default) is Median.
        level=25 is Q1.
        level=75 is Q3.
        """
        keyMap = {50: 'q', 25: 'q1', 75: 'q3'}
        return self._data.get(rank, {}).get(keyMap.get(level, 'q'), 0.0)

    def min(self, rank: int) -> float:
        """Returns the lower statistical whisker limit."""
        return self._data.get(rank, {}).get('min_w', 0.0)
        
    def max(self, rank: int) -> float:
        """Returns the upper statistical whisker limit."""
        return self._data.get(rank, {}).get('max_w', 0.0)

    def __repr__(self):
        return f"<CasteSummary: {list(self._data.keys())} ranks>"

class StratificationResult(StatsResult):
    """
    Results of a Population Stratification (Rank Distribution) analysis.
    
    Provides insights into the dominance structure of a population and 
    selection pressure.
    """
    def __init__(self, ranks_hist, raw_distributions=None, source=None, gen=-1, 
                 objectives=None, rank_array=None, sub_results=None):
        self.ranks = ranks_hist  # dict {rank: frequency}
        self.raw = raw_distributions # raw counts per run if applicable
        self.source = source
        self.gen = gen
        self.objectives = objectives # (N x M) objective matrix
        self.rank_array = rank_array # (N,) rank assigned to each row
        self.sub_results = sub_results # list of StratificationResult for sub-components
        
    @property
    def max_rank(self) -> int:
        return max(self.ranks.keys()) if self.ranks else 0

    def frequencies(self) -> np.ndarray:
        """Returns the distribution as a numpy array."""
        m = self.max_rank
        f = np.zeros(m)
        for r, val in self.ranks.items():
            f[r-1] = val
        return f

    def quality_profile(self) -> np.ndarray:
        """
        Calculates the average Quality (L2 norm) per rank.
        
        Returns:
            np.ndarray: Vector of average norms for each rank [1, 2, ...].
        """
        return self.quality_by(lambda x: np.mean(np.linalg.norm(x, axis=1)))

    def quality_by(self, metric_fn, **kwargs) -> np.ndarray:
        """
        Generic 'Map-Reduce' quality calculator per rank.
        
        Args:
            metric_fn: Callable that takes objective matrix (N x M) 
                       and returns a scalar quality measure.
            **kwargs: Passed to metric_fn.
            
        Returns:
            np.ndarray: Metric value for each rank [1, 2, ...].
        """
        if self.objectives is None or self.rank_array is None:
            return np.array([])
            
        m = self.max_rank
        scores = np.zeros(m)
        
        for r in range(1, m + 1):
            mask = (self.rank_array == r)
            if np.any(mask):
                scores[r-1] = metric_fn(self.objectives[mask], **kwargs)
            else:
                scores[r-1] = np.nan
        return scores

    @cached_property
    def selection_pressure(self) -> float:
        """
        Estimates the Selection Pressure as the exponential decay rate 
        of the rank distribution (Lazy calculation).
        """
        f = self.frequencies()
        if len(f) < 2: return 1.0 # Absolute pressure
        
        x = np.arange(len(f))
        y = np.log(f + 1e-10)
        res = linregress(x, y)
        return -res.slope

    def caste_summary(self, mode='individual', metric_fn=None, anchor=1.0, **m_kwargs) -> 'CasteSummary':
        """
        Extracts the exact statistical summary for each rank (as seen in strat_caste2).
        
        Args:
            mode (str): 'individual' or 'collective'.
            metric_fn: Metric to evaluate quality (default: hypervolume).
            anchor (float): Normalization constant.
            **m_kwargs: Passed to metric_fn.
            
        Returns:
            CasteSummary: Object with methods like .n(rank), .q(rank), .min(rank), .max(rank).
        """
        import MoeaBench.metrics.evaluator as eval_mod
        m_func = metric_fn if metric_fn else eval_mod.hypervolume
        
        stats = {}
        for r in range(1, self.max_rank + 1):
            mask = (self.rank_array == r)
            if not np.any(mask): continue
            
            sub_objs = self.objectives[mask]
            
            if mode == 'collective':
                if self.sub_results:
                    samples = []
                    for sub in self.sub_results:
                        s_mask = (sub.rank_array == r)
                        val = m_func(sub.objectives[s_mask], **m_kwargs) if np.any(s_mask) else 0.0
                        samples.append(float(val) / anchor)
                else:
                    samples = [float(m_func(sub_objs, **m_kwargs)) / anchor]
            else:
                samples = [float(m_func(sub_objs[j:j+1], **m_kwargs)) / anchor for j in range(len(sub_objs))]
                
            q_stats = np.percentile(samples, [25, 50, 75])
            q25, q50, q75 = q_stats
            iqr = q75 - q25
            
            up_w = q75 + 1.5 * iqr
            lo_w = q25 - 1.5 * iqr
            
            max_w = np.max([s for s in samples if s <= up_w]) if any(s <= up_w for s in samples) else q75
            min_w = np.min([s for s in samples if s >= lo_w]) if any(s >= lo_w for s in samples) else q25
            
            avg_n = np.mean([np.sum(s.rank_array == r) for s in self.sub_results]) if self.sub_results else len(sub_objs)
            
            stats[r] = {
                'n': int(round(avg_n)),
                'q': float(q50),
                'q1': float(q25),
                'q3': float(q75),
                'min_w': float(min_w),
                'max_w': float(max_w)
            }
        return CasteSummary(stats)

    def report(self, metric=None) -> str:
        """
        Generates an analytical narrative report of the population strata.
        """
        import MoeaBench.metrics.evaluator as eval_mod
        m_func = metric if metric else eval_mod.hypervolume
        
        name = getattr(self.source, 'name', 'Population')
        q_profile = self.quality_by(m_func)
        f = self.frequencies()
        
        lines = [
            f"--- Population Strata Report: {name} ---",
            f"  Search Depth: {self.max_rank} non-dominated layers",
            f"  Selection Pressure: {self.selection_pressure:.4f}",
            "",
            f"{'Rank':<6} | {'Pop %':<8} | {'Quality (' + m_func.__name__ + ')':<15}",
            "-" * 40
        ]
        
        for r_idx, freq in enumerate(f):
            r = r_idx + 1
            q = q_profile[r_idx]
            lines.append(f"{r:<6} | {freq*100:>7.1f}% | {q:>12.4f}")
            
        if self.selection_pressure > 0.8:
            lines.append("\nDiagnosis: High Selection Pressure (Phalanx-like convergence).")
        elif self.selection_pressure < 0.2:
            lines.append("\nDiagnosis: Low Selection Pressure (Exploratory/Scattered architecture).")
            
        return "\n".join(lines)

    def __repr__(self) -> str:
        name = getattr(self.source, 'name', 'Population')
        return f"<StratificationResult source='{name}' ranks={self.max_rank}>"

def strata(data, gen=-1):
    """
    Performs Population Stratification (Rank Distribution) analysis.
    """
    from ..core.run import Population, Run
    
    # helper to aggregate run data
    def _collect_run_data(run_obj, g):
        pop = run_obj.pop(g) if isinstance(run_obj, Run) else run_obj
        ranks = pop.stratify()
        hist = _calc_rank_hist(ranks)
        return hist, pop.objectives, ranks

    # 1. Handle Experiment (Multi-Run) or JoinedPopulation (Aggregated Snapshot)
    is_joined = False
    try:
        from ..core.experiment import JoinedPopulation
        if isinstance(data, JoinedPopulation):
            is_joined = True
    except ImportError:
        pass

    if (hasattr(data, 'runs') and hasattr(data, 'pop')) or is_joined:
        all_res = []
        
        # Determine iterator
        items = data.pops if is_joined else data
        
        for item in items:
            all_res.append(strata(item, gen))
            
        agg_hist = {}
        max_r = max(r.max_rank for r in all_res) if all_res else 0
        for r in range(1, max_r + 1):
            agg_hist[r] = np.mean([res.ranks.get(r, 0) for res in all_res])
            
        concat_objs = np.vstack([res.objectives for res in all_res])
        concat_ranks = np.concatenate([res.rank_array for res in all_res])
            
        return StratificationResult(agg_hist, raw_distributions=[res.ranks for res in all_res], 
                                    source=data, gen=gen, 
                                    objectives=concat_objs, rank_array=concat_ranks,
                                    sub_results=all_res)

    # 2. Handle Run/Population
    if isinstance(data, (Run, Population)):
        h, o, r = _collect_run_data(data, gen)
        return StratificationResult(h, source=data, gen=gen, objectives=o, rank_array=r)

    raise TypeError(f"Unsupported data type for stratification: {type(data)}")

def _calc_rank_hist(ranks):
    """Internal helper to calculate rank frequency distribution."""
    if len(ranks) == 0: return {}
    u, counts = np.unique(ranks, return_counts=True)
    total = len(ranks)
    return {int(r): c/total for r, c in zip(u, counts)}

def emd(strat1: StratificationResult, strat2: StratificationResult) -> float:
    """
    Computes the Earth Mover's Distance (Wasserstein Distance) between 
    two stratification profiles.
    
    This is a symmetric measure of how different the "dominance profiles" 
    of two algorithms are.
    """
    f1 = strat1.frequencies()
    f2 = strat2.frequencies()
    
    # Balance lengths
    m = max(len(f1), len(f2))
    v1 = np.pad(f1, (0, m - len(f1)))
    v2 = np.pad(f2, (0, m - len(f2)))
    
    val = wasserstein_distance(np.arange(m), np.arange(m), v1, v2)
    return SimpleStatsValue(val, "Structural Difference (EMD)")



class TierResult(StratificationResult):
    """
    Results of a Joint Stratification (Tier) analysis between two groups.
    
    Inherits from StratificationResult, adding support for relative 
    dominance comparison (proportions) between groups.
    """
    def __init__(self, agg_hist, raw_distributions=None, source=None, gen=-1, 
                 objectives=None, rank_array=None, group_labels=None, 
                 joint_frequencies=None, tier_counts=None):
        super().__init__(agg_hist, raw_distributions, source, gen, objectives, rank_array)
        self.group_labels = group_labels # list of names [A, B]
        self.joint_frequencies = joint_frequencies # dict {rank: [propA, propB]}
        self.tier_counts = tier_counts # dict {rank: absolute_count}

    @property
    def dominance_ratio(self) -> np.ndarray:
        """Returns the [A, B] proportion in the first rank (Elite)."""
        return self.joint_frequencies.get(1, np.array([0.5, 0.5]))

    @property
    def pole(self) -> np.ndarray:
        """F1 Metaphor: Returns the proportion in the first rank (Elite)."""
        return self.dominance_ratio

    @property
    def displacement_depth(self) -> int:
        """Finds the first rank where the relative proportion of B is > 0.1."""
        # This is a heuristic for when the 'loser' starts to appear.
        for r in range(1, self.max_rank + 1):
            if self.joint_frequencies[r][1] > 0.1:
                return r
        return self.max_rank

    @property
    def gap(self) -> int:
        """F1 Metaphor: Displacement depth (where the rival starts to appear)."""
        return self.displacement_depth

    def report(self, metric=None) -> str:
        """Generates a competitive narrative report using Tier/F1 metaphors."""
        nameA, nameB = self.group_labels
        ratioA, ratioB = self.pole
        
        lines = [
            f"--- Competitive Tier Report: {nameA} vs {nameB} ---",
            f"  Search Depth: {self.max_rank} global levels",
            f"  Dominance Ratio (Tier 1): {nameA} ({ratioA*100:.1f}%) | {nameB} ({ratioB*100:.1f}%)",
            f"  Displacement Depth: {self.gap} (Rank where rival > 10%)",
            "",
            f"{'Tier':<6} | {nameA + ' %':<10} | {nameB + ' %':<10}",
            "-" * 35
        ]
        
        for r in range(1, self.max_rank + 1):
            propA, propB = self.joint_frequencies.get(r, [0, 0])
            lines.append(f"{r:<6} | {propA*100:>8.1f}% | {propB*100:>8.1f}%")
            
        better = nameA if ratioA > 0.5 else nameB
        lines.append(f"\nDiagnosis: {better} occupies the Pole Position.")
        if self.gap > 2:
            lines.append(f"Observation: {nameA if better == nameA else nameB} significantly 'buries' the rival (Large Gap > 2 ranks).")
            
        return "\n".join(lines)

def tier(exp1, exp2, gen=-1):
    """
    Performs Joint Stratification (Tier Analysis) between two experiments.
    """
    from ..core.run import SmartArray
    # 1. Collect all final fronts (or specific gen)
    def _get_objs_with_src(exp, label):
        objs = []
        if hasattr(exp, 'runs'):
            for run in exp:
                objs.append(run.front(gen))
        else:
            objs.append(exp.objectives)
        return np.vstack(objs), np.full(sum(len(o) for o in objs), label)

    objsA, labelsA = _get_objs_with_src(exp1, 0)
    objsB, labelsB = _get_objs_with_src(exp2, 1)
    
    combined_objs = np.vstack([objsA, objsB])
    combined_labels = np.concatenate([labelsA, labelsB])
    
    # 2. Perform Global NDS
    from ..core.run import Population
    pop = Population(combined_objs, combined_objs)
    global_ranks = pop.stratify()
    
    # 3. Calculate Joint Frequencies
    unique_ranks = np.unique(global_ranks)
    joint_freqs = {}
    agg_hist = {}
    agg_counts = {}
    
    for r in unique_ranks:
        mask = (global_ranks == r)
        total_in_rank = np.sum(mask)
        # Proportions of A and B in this rank
        propA = np.sum((combined_labels[mask] == 0)) / total_in_rank
        propB = np.sum((combined_labels[mask] == 1)) / total_in_rank
        joint_freqs[int(r)] = np.array([propA, propB])
        agg_hist[int(r)] = total_in_rank / len(combined_objs)
        agg_counts[int(r)] = int(total_in_rank)
        
    nameA = getattr(exp1, 'name', 'Algorithm A')
    nameB = getattr(exp2, 'name', 'Algorithm B')
    
    return TierResult(agg_hist, source=(exp1, exp2), gen=gen, 
                       objectives=combined_objs, rank_array=global_ranks,
                       group_labels=[nameA, nameB],
                       joint_frequencies=joint_freqs,
                       tier_counts=agg_counts)
