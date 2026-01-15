# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from scipy.stats import linregress, wasserstein_distance

class StratificationResult:
    """
    Results of a Population Stratification (Rank Distribution) analysis.
    
    Provides insights into the dominance structure of a population and 
    selection pressure.
    """
    def __init__(self, ranks_hist, raw_distributions=None, source=None, gen=-1, 
                 objectives=None, rank_array=None):
        self.ranks = ranks_hist  # dict {rank: frequency}
        self.raw = raw_distributions # raw counts per run if applicable
        self.source = source
        self.gen = gen
        self.objectives = objectives # (N x M) objective matrix
        self.rank_array = rank_array # (N,) rank assigned to each row
        
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

    def selection_pressure(self) -> float:
        """
        Estimates the Selection Pressure as the exponential decay rate 
        of the rank distribution.
        """
        f = self.frequencies()
        if len(f) < 2: return 1.0 # Absolute pressure
        
        x = np.arange(len(f))
        y = np.log(f + 1e-10)
        res = linregress(x, y)
        return -res.slope

    def plot(self, title=None, **kwargs):
        """Generates a bar plot of the rank distribution."""
        import matplotlib.pyplot as plt
        f = self.frequencies()
        plt.bar(np.arange(1, len(f) + 1), f, **kwargs)
        plt.xlabel("Dominance Rank (1=Pareto Front)")
        plt.ylabel("Frequency (%)")
        if title: plt.title(title)
        return plt.gca()

    def __repr__(self) -> str:
        name = getattr(self.source, 'name', 'Population')
        return f"<StratificationResult source='{name}' ranks={self.max_rank}>"

def stratification(data, gen=-1):
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

    # 1. Handle Experiment (Multi-Run)
    if hasattr(data, 'runs') and hasattr(data, 'pop'):
        all_hists = []
        all_objs = []
        all_ranks = []
        for run in data:
            h, o, r = _collect_run_data(run, gen)
            all_hists.append(h)
            all_objs.append(o)
            all_ranks.append(r)
            
        agg_hist = {}
        max_r = max(max(h.keys()) for h in all_hists)
        for r in range(1, max_r + 1):
            agg_hist[r] = np.mean([h.get(r, 0) for h in all_hists])
            
        # For aggregated quality, we'll keep the raw components to allow 
        # StratificationResult to calculate means correctly.
        # However, for simplicity and memory, we can store the concatenated 
        # objectives and ranks as if it were one giant run for the quality profile.
        concat_objs = np.vstack(all_objs)
        concat_ranks = np.concatenate(all_ranks)
            
        return StratificationResult(agg_hist, raw_distributions=all_hists, 
                                    source=data, gen=gen, 
                                    objectives=concat_objs, rank_array=concat_ranks)

    # 2. Handle Run/Population
    if isinstance(data, (Run, Population)):
        h, o, r = _collect_run_data(data, gen)
        return StratificationResult(h, source=data, gen=gen, objectives=o, rank_array=r)

    raise TypeError(f"Unsupported data type for stratification: {type(data)}")

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
    
    return wasserstein_distance(np.arange(m), np.arange(m), v1, v2)

def stratification_plot(*results: StratificationResult, labels: list = None, title: str = None, **kwargs):
    """
    Generates a grouped bar chart to compare multiple stratification profiles.
    
    Args:
        *results: One or more StratificationResult objects.
        labels (list): Optional labels for the legend.
        title (str): Plot title.
        **kwargs: Passed to plt.bar.
    """
    import matplotlib.pyplot as plt
    
    if not results:
        return
        
    m = max(r.max_rank for r in results)
    ranks = np.arange(1, m + 1)
    n_series = len(results)
    width = 0.8 / n_series
    
    fig, ax = plt.subplots()
    
    for i, res in enumerate(results):
        f = res.frequencies()
        # Pad frequencies to match max rank
        f_padded = np.pad(f, (0, m - len(f)))
        
        lbl = labels[i] if labels and i < len(labels) else getattr(res.source, 'name', f"Series {i+1}")
        
        pos = ranks - 0.4 + (i + 0.5) * width
        ax.bar(pos, f_padded, width, label=lbl, **kwargs)

    ax.set_xlabel("Dominance Rank (1=Pareto Front)")
    ax.set_ylabel("Frequency (%)")
    ax.set_xticks(ranks)
    ax.legend()
    if title: ax.set_title(title)
    
    return ax

def _calc_rank_hist(ranks):
    """
    Helper to convert rank array to frequency distribution.
    
    Note: Ranks are 1-indexed. Rank 1 is the Pareto Front. 
    Rank 2 is the Pareto Front of the population remaining after 
    removing Rank 1, and so on.
    """
    if len(ranks) == 0: return {}
    unique, counts = np.unique(ranks, return_counts=True)
    freqs = counts / len(ranks)
    return dict(zip(unique.astype(int), freqs))
