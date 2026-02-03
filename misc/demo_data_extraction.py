import MoeaBench as mb
import numpy as np

def demo_extraction():
    # 1. Setup Data
    exp = mb.experiment()
    exp.mop = mb.mops.DTLZ2(M=3)
    exp.moea = mb.moeas.NSGA3(population=100)
    exp.run(repeat=3)
    
    # 2. Extracting "Caste/Rank" Data
    # The view (strat_ranks, strat_caste) uses mb.stats.strata internally
    res = mb.stats.strata(exp.pop(50))
    
    print("--- Rank Distribution (Used in strat_ranks) ---")
    print(f"Frequencies array: {res.frequencies()}")
    print(f"Max rank attained: {res.max_rank}")
    
    print("\n--- Quality Profile (Used in strat_caste) ---")
    # Standard profile (average HV per rank)
    q_profile = res.quality_profile()
    for r, q in enumerate(q_profile, 1):
        print(f"Rank {r}: Mean Quality = {q:.4f}")
        
    print("\n--- Programmatic Caste Summary (Method-based API) ---")
    # Returns a CasteSummary object with intuitive methods (.n, .q, .min, .max)
    summary = res.caste_summary(mode='individual', anchor=1.0)
    
    # Access Rank 1 stats using methods instead of dictionaries
    print(f"Rank 1 [Individual]: n={summary.n(1)}, Median={summary.q(1):.2f}, Q1={summary.q(1, 25):.2f}, Q3={summary.q(1, 75):.2f}, Max Whisker={summary.max(1):.2f}")
    
    # Collective view summary
    summary_coll = res.caste_summary(mode='collective', anchor=1.0)
    print(f"Rank 1 [Collective]: n={summary_coll.n(1)}, Median={summary_coll.q(1):.2f}, Robustness Range=[{summary_coll.min(1):.2f}, {summary_coll.max(1):.2f}]")

    # 3. Extracting "Tier" Data (Competitions)
    exp2 = mb.experiment()
    exp2.mop = mb.mops.DTLZ2(M=3)
    exp2.moea = mb.moeas.SPEA2(population=100)
    exp2.run(repeat=3)
    
    t_res = mb.stats.tier(exp, exp2)
    
    print("\n--- Competitive Metrics (Used in strat_tiers) ---")
    print(f"Dominance Ratio (Elite): {t_res.dominance_ratio}") # [PropA, PropB]
    print(f"Displacement Depth (Gap): {t_res.gap}") # The "F1 Gap" metaphor
    print(f"Search Depth: {t_res.max_rank}")

if __name__ == "__main__":
    demo_extraction()
