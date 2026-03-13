# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from ..defaults import defaults
from scipy.stats import linregress, wasserstein_distance
from functools import cached_property
from typing import Dict, Any
from .base import StatsResult, SimpleStatsValue

class StrataSummary:
    """
    Helper object to access strata statistics using intuitive methods.
    """
    def __init__(self, data: Dict[int, Dict], name: str = "Population", mode: str = "individual", metric_name: str = "hypervolume"):
        self._data = data
        self.name = name
        self.mode = mode
        self.metric_name = metric_name
        
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
        return f"<StrataSummary {self.name}: {list(self._data.keys())} ranks>"


class RankCompareResult(StatsResult):
    """Canonical result for rank-structure analysis."""

    def __init__(self, results, labels):
        self.results = results
        self.labels = labels

    def report(self, show: bool = True, **kwargs) -> str:
        use_md = kwargs.get("markdown", self._is_notebook())
        if use_md:
            lines = ["### Rank Structure Report", "", "| Data | Depth | Pressure |", "| :--- | ---: | ---: |"]
            for label, res in zip(self.labels, self.results):
                lines.append(f"| {label} | {res.max_rank} | {res.selection_pressure:.4f} |")
            content = "\n".join(lines)
        else:
            lines = ["Rank Structure Report", "", f"{'Data':<16} | {'Depth':>5} | {'Pressure':>8}", "-" * 38]
            for label, res in zip(self.labels, self.results):
                lines.append(f"{label:<16} | {res.max_rank:>5} | {res.selection_pressure:>8.4f}")
            content = "\n".join(lines)
        return self._render_report(content, show, **kwargs)


class StrataCompareResult(StatsResult):
    """Canonical result for strata-distribution analysis."""

    def __init__(self, summaries, labels, mode="collective", metric_name="hypervolume"):
        self.summaries = summaries
        self.labels = labels
        self.mode = mode
        self.metric_name = metric_name

    def report(self, show: bool = True, **kwargs) -> str:
        use_md = kwargs.get("markdown", self._is_notebook())
        rank_ids = sorted({rank for summary in self.summaries for rank in summary._data.keys()})

        if use_md:
            lines = [
                f"### Strata Distribution Report ({self.metric_name})",
                "",
                f"**Mode**: {self.mode}",
                "",
                "| Data | Rank | n | Q1 | Median | Q3 |",
                "| :--- | ---: | ---: | ---: | ---: | ---: |",
            ]
            for label, summary in zip(self.labels, self.summaries):
                for rank in rank_ids:
                    info = summary._data.get(rank)
                    if not info:
                        continue
                    lines.append(
                        f"| {label} | {rank} | {info['n']} | {info['q1']:.4f} | "
                        f"{info['q']:.4f} | {info['q3']:.4f} |"
                    )
            content = "\n".join(lines)
        else:
            lines = [
                f"Strata Distribution Report ({self.metric_name})",
                f"Mode: {self.mode}",
                "",
                f"{'Data':<16} | {'Rank':>4} | {'n':>4} | {'Q1':>8} | {'Median':>8} | {'Q3':>8}",
                "-" * 66,
            ]
            for label, summary in zip(self.labels, self.summaries):
                for rank in rank_ids:
                    info = summary._data.get(rank)
                    if not info:
                        continue
                    lines.append(
                        f"{label:<16} | {rank:>4} | {info['n']:>4} | {info['q1']:>8.4f} | "
                        f"{info['q']:>8.4f} | {info['q3']:>8.4f}"
                    )
            content = "\n".join(lines)
        return self._render_report(content, show, **kwargs)

class LayerResult(StatsResult):
    """
    Results of a population layer analysis based on dominance depth.
    
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
        self.sub_results = sub_results # list of LayerResult for sub-components
        
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

    def strata_summary(self, mode='individual', metric_fn=None, anchor=1.0, **m_kwargs) -> 'StrataSummary':
        """
        Extracts the exact statistical summary for each rank layer.
        
        Args:
            mode (str): 'individual' or 'collective'.
            metric_fn: Metric to evaluate quality (default: hypervolume).
            anchor (float): Normalization constant.
            **m_kwargs: Passed to metric_fn.
            
        Returns:
            StrataSummary: Object with methods like .n(rank), .q(rank), .min(rank), .max(rank).
        """
        import moeabench.metrics.evaluator as eval_mod
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
        name = getattr(self.source, 'name', 'Population')
        metric_name = getattr(m_func, '__name__', 'hypervolume')
        return StrataSummary(stats, name=name, mode=mode, metric_name=metric_name)

    def report(self, show: bool = True, **kwargs) -> str:
        """
        Generates an analytical narrative report of the population layer structure.
        """
        use_md = kwargs.get('markdown', self._is_notebook())
        metric = kwargs.get('metric', None)
        import moeabench.metrics.evaluator as eval_mod
        m_func = metric if metric else eval_mod.hypervolume
        
        name = getattr(self.source, 'name', 'Population')
        q_profile = self.quality_by(m_func)
        f = self.frequencies()
        
        if use_md:
            header = f"### Population Layer Report: {name}"
            lines = [
                header,
                f"**Search Depth**: {self.max_rank} non-dominated layers",
                f"**Selection Pressure**: {self.selection_pressure:.4f}",
                "",
                "| Rank | Pop % | Quality (" + m_func.__name__ + ") |",
                "| :--- | :--- | :--- |"
            ]
            for r_idx, freq in enumerate(f):
                r = r_idx + 1
                q = q_profile[r_idx]
                lines.append(f"| {r} | {freq*100:.1f}% | {q:.4f} |")
            
            if self.selection_pressure > 0.8:
                lines.append("\n> **Diagnosis**: High Selection Pressure (Phalanx-like convergence).")
            elif self.selection_pressure < 0.2:
                lines.append("\n> **Diagnosis**: Low Selection Pressure (Exploratory/Scattered architecture).")
            
            content = "\n".join(lines)
        else:
            lines = [
                f"--- Population Layer Report: {name} ---",
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
            
            content = "\n".join(lines)
            
        return self._render_report(content, show, **kwargs)

    def __repr__(self) -> str:
        name = getattr(self.source, 'name', 'Population')
        return f"<LayerResult source='{name}' ranks={self.max_rank}>"

def _joint_layer_result(data_a, data_b, gen=-1):
    """Build a joint layer result used by tier visualizations."""
    from ..core.experiment import experiment
    from ..core.run import Run, Population

    def _extract_objs(data):
        if isinstance(data, LayerResult):
            if data.objectives is None:
                raise ValueError("LayerResult without objectives cannot be used in joint layers.")
            return np.asarray(data.objectives), getattr(data.source, "name", None) or getattr(data, "name", None)
        if isinstance(data, experiment):
            objs = [run.front(gen) for run in data]
            return np.vstack(objs), getattr(data, "name", "Algorithm")
        if isinstance(data, Run):
            pop = data.pop(gen)
            return np.asarray(pop.objectives), getattr(data, "name", "Run")
        if isinstance(data, Population):
            return np.asarray(data.objectives), getattr(data, "name", "Population")
        if isinstance(data, np.ndarray):
            return np.asarray(data), "Array"
        if hasattr(data, "objectives"):
            return np.asarray(data.objectives), getattr(data, "name", "Data")
        raise TypeError(f"Unsupported data type for joint layers: {type(data)}")

    objs_a, name_a = _extract_objs(data_a)
    objs_b, name_b = _extract_objs(data_b)

    combined_objs = np.vstack([objs_a, objs_b])
    labels = np.concatenate([np.zeros(len(objs_a), dtype=int), np.ones(len(objs_b), dtype=int)])

    from ..core.run import Population
    pop = Population(combined_objs, combined_objs)
    global_ranks = pop.stratify()

    unique_ranks = np.unique(global_ranks)
    joint_freqs = {}
    agg_hist = {}
    agg_counts = {}

    for r in unique_ranks:
        mask = (global_ranks == r)
        total = np.sum(mask)
        prop_a = np.sum(labels[mask] == 0) / total
        prop_b = np.sum(labels[mask] == 1) / total
        joint_freqs[int(r)] = np.array([prop_a, prop_b])
        agg_hist[int(r)] = total / len(combined_objs)
        agg_counts[int(r)] = int(total)

    return TierResult(
        agg_hist,
        source=(data_a, data_b),
        gen=gen,
        objectives=combined_objs,
        rank_array=global_ranks,
        group_labels=[name_a or "Algorithm A", name_b or "Algorithm B"],
        joint_frequencies=joint_freqs,
        tier_counts=agg_counts,
    )


def _layer(data, other=None, gen=-1):
    """
    Performs dominance-layer analysis.
    """
    from ..core.run import Population, Run

    if other is not None:
        return _joint_layer_result(data, other, gen=gen)

    if isinstance(data, LayerResult):
        return data
    
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
            all_res.append(_layer(item, gen=gen))
            
        agg_hist = {}
        max_r = max(r.max_rank for r in all_res) if all_res else 0
        for r in range(1, max_r + 1):
            agg_hist[r] = np.mean([res.ranks.get(r, 0) for res in all_res])
            
        concat_objs = np.vstack([res.objectives for res in all_res])
        concat_ranks = np.concatenate([res.rank_array for res in all_res])
            
        return LayerResult(
            agg_hist,
            raw_distributions=[res.ranks for res in all_res],
            source=data,
            gen=gen,
            objectives=concat_objs,
            rank_array=concat_ranks,
            sub_results=all_res,
        )

    # 2. Handle Run/Population
    if isinstance(data, (Run, Population)):
        h, o, r = _collect_run_data(data, gen)
        return LayerResult(h, source=data, gen=gen, objectives=o, rank_array=r)

    # 3. Handle Raw Objective Matrices (np.ndarray, SmartArray, list)
    if isinstance(data, (np.ndarray, list, tuple)):
        # Convert to Population to reuse stratification logic
        pop = Population(np.asarray(data))
        ranks = pop.stratify()
        hist = _calc_rank_hist(ranks)
        return LayerResult(hist, source=data, gen=gen, objectives=pop.objectives, rank_array=ranks)

    raise TypeError(f"Unsupported data type for layer analysis: {type(data)}")

def _calc_rank_hist(ranks):
    """Internal helper to calculate rank frequency distribution."""
    if len(ranks) == 0: return {}
    u, counts = np.unique(ranks, return_counts=True)
    total = len(ranks)
    return {int(r): c/total for r, c in zip(u, counts)}

def emd(strat1: LayerResult, strat2: LayerResult) -> float:
    """
    Computes the Earth Mover's Distance (Wasserstein Distance) between 
    two layer profiles.
    
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



class TierResult(LayerResult):
    """
    Results of a joint layer (tier) analysis between two groups.
    
    Inherits from LayerResult, adding support for relative
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
        """Finds the first rank where the relative proportion of B is > threshold."""
        # This is a heuristic for when the 'loser' starts to appear.
        threshold = defaults.displacement_threshold
        for r in range(1, self.max_rank + 1):
            if self.joint_frequencies[r][1] > threshold:
                return r
        return self.max_rank

    @property
    def gap(self) -> int:
        """F1 Metaphor: Displacement depth (where the rival starts to appear)."""
        return self.displacement_depth

    def report(self, show: bool = True, **kwargs) -> str:
        """Generates a competitive narrative report using Tier/F1 metaphors."""
        use_md = kwargs.get('markdown', self._is_notebook())
        nameA, nameB = self.group_labels
        ratioA, ratioB = self.pole
        
        if use_md:
            header = f"### Competitive Tier Report: {nameA} vs {nameB}"
            lines = [
                header,
                f"**Search Depth**: {self.max_rank} global levels",
                f"**Dominance Ratio (Tier 1)**: {nameA} ({ratioA*100:.1f}%) | {nameB} ({ratioB*100:.1f}%)",
                f"**Displacement Depth**: {self.gap} (Rank where rival > {defaults.displacement_threshold*100:.0f}%)",
                "",
                f"| Tier | {nameA} % | {nameB} % |",
                "| :--- | :--- | :--- |"
            ]
            for r in range(1, self.max_rank + 1):
                propA, propB = self.joint_frequencies.get(r, [0, 0])
                lines.append(f"| {r} | {propA*100:.1f}% | {propB*100:.1f}% |")
            
            better = nameA if ratioA > 0.5 else nameB
            lines.append(f"\n> **Diagnosis**: **{better}** occupies the Pole Position.")
            if self.gap > defaults.large_gap_threshold:
                lines.append(f"> **Observation**: {better} significantly 'buries' the rival (Large Gap > {defaults.large_gap_threshold} ranks).")
            
            content = "\n".join(lines)
        else:
            lines = [
                f"--- Competitive Tier Report: {nameA} vs {nameB} ---",
                f"  Search Depth: {self.max_rank} global levels",
                f"  Dominance Ratio (Tier 1): {nameA} ({ratioA*100:.1f}%) | {nameB} ({ratioB*100:.1f}%)",
                f"  Displacement Depth: {self.gap} (Rank where rival > {defaults.displacement_threshold*100:.0f}%)",
                "",
                f"{'Tier':<6} | {nameA + ' %':<10} | {nameB + ' %':<10}",
                "-" * 35
            ]
            for r in range(1, self.max_rank + 1):
                propA, propB = self.joint_frequencies.get(r, [0, 0])
                lines.append(f"{r:<6} | {propA*100:>8.1f}% | {propB*100:>8.1f}%")
            
            better = nameA if ratioA > 0.5 else nameB
            lines.append(f"\nDiagnosis: {better} occupies the Pole Position.")
            if self.gap > defaults.large_gap_threshold:
                lines.append(f"Observation: {better} significantly 'buries' the rival (Large Gap > {defaults.large_gap_threshold} ranks).")
            
            content = "\n".join(lines)
            
        return self._render_report(content, show, **kwargs)


def ranks(*args, gen=-1):
    """Canonical rank-structure analysis result."""
    if len(args) == 1 and isinstance(args[0], RankCompareResult):
        return args[0]

    results = []
    labels = []
    for arg in args:
        res = _layer(arg, gen=gen)
        results.append(res)
        label = getattr(res.source, "name", None) or getattr(arg, "name", None) or "Population"
        labels.append(label)
    return RankCompareResult(results, labels)


def strata(*args, metric=None, mode='collective', gen=-1, **kwargs):
    """Canonical strata-distribution analysis result."""
    import moeabench.metrics.evaluator as eval_mod

    if len(args) == 1 and isinstance(args[0], StrataCompareResult):
        return args[0]

    metric_fn = metric or eval_mod.hypervolume
    results = [_layer(arg, gen=gen) for arg in args]
    labels = [getattr(res.source, "name", None) or getattr(arg, "name", None) or "Population"
              for res, arg in zip(results, args)]

    all_objs_list = [res.objectives for res in results if res.objectives is not None]
    global_ref = np.vstack(all_objs_list) if all_objs_list else None
    total_qualities = []
    for res in results:
        if res.objectives is not None:
            val = metric_fn(res.objectives, ref=global_ref, **kwargs)
            total_qualities.append(float(val))
    anchor = max(total_qualities) if total_qualities else 1.0

    summaries = [
        res.strata_summary(mode=mode, metric_fn=metric_fn, anchor=anchor, ref=global_ref, **kwargs)
        for res in results
    ]
    metric_name = getattr(metric_fn, "__name__", "hypervolume")
    return StrataCompareResult(summaries, labels, mode=mode, metric_name=metric_name)


def tiers(data, other=None, gen=-1):
    """Canonical tier-duel analysis result."""
    if isinstance(data, TierResult) and other is None:
        return data
    if other is None:
        raise ValueError("mb.stats.tiers requires two inputs or a TierResult.")
    return _layer(data, other, gen=gen)
