import MoeaBench as mb
exp = mb.experiment()
exp.mop = mb.mops.DPF2(M=3)
exp.moea = mb.moeas.NSGA2(population=52, generations=2)
exp.run(quiet=True)
run = exp.runs[0]
print(hasattr(exp, 'pop'))
print(hasattr(exp, 'population'))
print(hasattr(run, 'last_pop'))
for attr in dir(run):
  if 'pop' in attr:
      print("run."+attr)
