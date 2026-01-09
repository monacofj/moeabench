from MoeaBench import mb
import os, importlib
import numpy as np


os.system("cls")  





exp7  = mb.experiment()
exp7.load("repo_up_2")
hv  = mb.hypervolume(exp7) 



exp7.benchmark.M = 5

exp7.run()
exp7.benchmark.D = 3
exp7.benchmark.M = 4
exp7.run(repeat = 3)
exp7.save("test11")
exp8 = mb.experiment()
exp8.load("test11")
exp8.benchmark.D = 2
exp8.benchmark.M = 3
exp8.run(repeat = 5)