import os
from MoeaBench import mb
from MoeaBench.my_dtlz5 import my_dtlz5
from MoeaBench.NSGA2deap import my_NSGA2deap



os.system("cls")  



mb.add_benchmark(my_dtlz5)
mb.add_moea(my_NSGA2deap)



exp = mb.experiment()
exp.save("init")

exp3 = mb.experiment()
exp3.load("init")

exp3.benchmark = mb.benchmarks.my_dtlz5()
exp3.save("init2")

exp4 = mb.experiment()
exp4.load("init2")
exp4.moea = mb.moeas.my_NSGA2deap()

exp4.save("init3")

exp5 = mb.experiment()
exp5.load("init3")
exp5.benchmark.M = 4



exp5.save("init4")

exp6 = mb.experiment()
exp6.load("init4")

exp6.run()

exp6.benchmark = mb.benchmarks.DPF5()
exp6.moea = mb.moeas.SPEA2()
exp6.run()
exp6.save("repo")

exp6.benchmark.M=5



exp6.benchmark.K=10


exp6.benchmark.P=2000


exp6.benchmark.D=3


exp6.moea.population = 200
exp6.moea.generations = 400
exp6.run()


exp6.save("repo_up")

exp6.benchmark.M=5
exp6.benchmark.D=4
exp6.run(repeat = 2)


exp6.save("repo_up_2")

exp7  = mb.experiment()
exp7.load("repo_up_2")

