#!/bin/env python3

import mb_path
from moeabench import mb

# Dois experimentos
exp1 = mb.experiment(mop=mb.mops.DTLZ2(M=3), moea=mb.moeas.NSGA2(population=60, generations=50))
exp2 = mb.experiment(mop=mb.mops.DTLZ2(M=3), moea=mb.moeas.NSGA3(population=60, generations=50))

exp1.run(repeat=5, silent=True)
exp2.run(repeat=5, silent=True)

# Métricas (MetricMatrix)
hv1 = mb.metrics.hv(exp1, ref=[exp1, exp2])
hv2 = mb.metrics.hv(exp2, ref=[exp1, exp2])

# Uso público direto do plot_matrix
mb.metrics.plot_matrix(
    [hv1, hv2],
    title="Hypervolume Comparison",
    mode="auto",        # auto | static | interactive
    show_bounds=True
)
