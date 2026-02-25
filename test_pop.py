import MoeaBench as mb
exp = mb.experiment()
exp.mop = mb.mops.DPF2(M=3)
exp.moea = mb.moeas.NSGA2(population=50, generations=2)
exp.run(quiet=True)
print(type(exp.runs[0].evals))
print(exp.runs[0].evals.shape)
