from MoeaBench import mb
import os, importlib
import numpy as np


os.system("cls")  


exp = mb.experiment()
exp.name = 'experiment 1'
exp.benchmark = mb.benchmarks.DTLZ1()
exp.moea = mb.moeas.NSGA3(generations = 10, population = 150)
exp.moea.generations=150
exp.moea.seed = 4
exp.name = "turicer"
exp.run(repeat= 2)

#var = exp.variables(generation = 1)
#print(var.shape)

#ar_r = exp.variables.round(1)
#print(var_r.shape)
#set = exp.set(generation = 89)
#print(exp.set.round(1))


print(exp.dominated.objectives(generation = 3).shape)
print(exp.dominated.objectives.round(1).shape)


print(exp.dominated.variables(generation = 3).shape)
print(exp.dominated.variables.round(1).shape)

pr = mb.stats.paretorank(exp)
print(pr.rank())

#var = exp.dominated.variables(generation = 89)
#@print(var.shape)


#for i in range(0,4):
  #obj_r = exp.set.round(i)
 # print(obj_r.shape)


#print(exp.rounds[1].variables)