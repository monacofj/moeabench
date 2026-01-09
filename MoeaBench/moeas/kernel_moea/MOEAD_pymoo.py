from pymoo.optimize import minimize
import numpy as np
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.decomposition.pbi import PBI
from pymoo.core.callback import Callback


class callback_stop(Callback):

    def __init__(self, stop, experiment, name):
        super().__init__()
        self.stop = stop
        self.name = name
        self.experiment = experiment
        self.arr_f_pop = []
        self.arr_x_pop = []
        self.arr_f_nd = []
        self.arr_x_nd = []
        self.arr_f_dom = []
        self.arr_x_dom = []


    def save(self, gen, pop):
        problem = self.experiment.result.edit_DATA_conf().get_problem()
        name_benchmark = problem.__class__.__name__.split("_")[1]
        self.experiment.result.DATA_store(
                              self.name, 
                              gen,
                              pop,
                              self.arr_f_pop[-1],
                              self.arr_f_pop,
                              self.arr_x_pop,
                              problem,
                              name_benchmark,
                              self.arr_f_nd,
                              self.arr_x_nd,
                              self.arr_f_dom, 
                              self.arr_x_dom
                              )

    
    def notify(self, algorithm):

        f_pop = algorithm.pop.get("F")
        x_pop = algorithm.pop.get("X")
        f_nd = algorithm.opt.get("F")
        x_nd = algorithm.opt.get("X")

        nd_set_f = set(map(tuple, f_nd))
        ref_dom_f = [tuple(f) not in nd_set_f for f in f_pop]
        f_dom = f_pop[ref_dom_f]     

        nd_set_x = set(map(tuple, x_nd))
        ref_dom_x = [tuple(x) not in nd_set_x for x in x_pop]
        x_dom = x_pop[ref_dom_x]

        self.arr_f_pop.append(f_pop)
        self.arr_x_pop.append(x_pop)
        self.arr_f_nd.append(f_nd)
        self.arr_x_nd.append(x_nd)
        self.arr_f_dom.append(f_dom)
        self.arr_x_dom.append(x_dom)

        self.save(algorithm.n_gen, len(algorithm.pop))
        if  callable(self.stop) and self.stop(self.experiment):
            algorithm.termination.force_termination = True


class MOEAD_pymoo(Problem):
    
    def __init__(self,experiment,population,generations,seed,stop):
        self.experiment=experiment
        self.population=population
        self.generations=generations
        self.seed=int(seed.integers(low= 0 , high = 100, size = 1)[0]) if not isinstance(seed, int) else seed
        self.stop=stop
        self.Nvar=self.experiment.benchmark.get_CACHE().get_BENCH_CI().get_Nvar()
        self.M=self.experiment.benchmark.get_CACHE().get_BENCH_CI().get_M()
        self.BENCH_Nvar=self.experiment.benchmark.get_CACHE().get_BENCH_CI().get_BENCH_Nvar()
        xl = np.full(self.Nvar,0)
        xu = np.full(self.Nvar,1)
        self.objectives = self.M
        super(). __init__(n_var=self.Nvar, n_obj=  self.objectives, xl=xl, xu=xu)

         
    def _evaluate(self, x, out, *args, **kwargs):  
        result = self.experiment.benchmark.evaluation(x,0)
        out["F"]=result['F']
        if "G" in result:
            out["G"]=result['G'] 
        

    def exec(self):
        stopping = callback_stop(self.stop, self.experiment, "MOEAD")
        ref_dirs = get_reference_directions("energy", self.objectives, self.population, seed = self.seed)  
        muttation_prob = 1/self.Nvar
        muttation=PolynomialMutation(prob=muttation_prob, eta = 20)
        crossover = SBX(prob=1.0, eta=15)
        algorithm_MOEAD = MOEAD(ref_dirs, crossover=crossover,mutation=muttation, decomposition=PBI(eps=0.0, theta=5))      
        res_MOEAD = minimize(
            MOEAD_pymoo(self.experiment,self.population, self.generations,self.seed,self.stop),
            algorithm_MOEAD,
            termination=('n_gen', self.generations),
    
            seed=self.seed,
            save_history=True,
            verbose=False,
            callback=stopping
            )          
       
    
