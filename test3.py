from MoeaBench.base_benchmark import BaseBenchmark
from enum import Enum
import numpy as np
import os
from MoeaBench import mb




os.system("cls")  






class E_DTLZ(Enum):
       F1   = 1
       F2   = 2
       F3   = 3 
       Fm   = 5


@mb.benchmarks.register_benchmark()
class dtlz5(BaseBenchmark):

    def __init__(self, types : str = None, M : int = 3, P : int = 700, K : int = 10, N : int = 0, D : int = 2, n_ieq_constr : int = 1):
        super().__init__(types, M, P, K, self.calc_N, n_ieq_constr)
        self.llist_E_DTLZ = list(E_DTLZ)
        

    def calc_N(self,K,M):
        return K+M-1


    def constraits(self,f,parameter = 1,f_c=[]):
        f_constraits=np.array(f)
        f_c = np.array([np.sum([ f_c**2  for  f_c in f_constraits[linha,0:f_constraits.shape[1]]])-parameter for index,linha in enumerate(range(f_constraits.shape[0]))  ])
        return f_c


    def eval_cons(self,f):
        M_constraits = self.constraits(f)
        eval = M_constraits == 0
        return f[eval]


    def get_Points(self):
        return np.array([*np.random.random((self.get_P(), self.get_N()))*1.0])


    def F1(self,M,th,Gxm):
       theta = list(map(lambda TH: np.cos(TH), th[0:(M-1)]))
       return (1+Gxm)*np.prod(np.column_stack(theta ), axis = 1).reshape(Gxm.shape[0],1)


    def F2(self,M,th,Gxm):
        theta = list(map(lambda TH: np.cos(TH), th[0:(M-2)]))
        return (1+Gxm)*np.prod(np.column_stack(theta ), axis = 1).reshape(Gxm.shape[0],1)*np.column_stack(np.sin(th[(M-2):(M-1)]))


    def F3(self,M,th,Gxm):
        theta = list(map(lambda TH: np.cos(TH), th[0:(M-3)]))
        return (1+Gxm)*np.prod(np.column_stack(theta ), axis = 1).reshape(Gxm.shape[0],1)*np.column_stack(np.sin(th[(M-3):(M-2)]))


    def Fm(self,M,th,Gxm):
        return (1+Gxm)*np.column_stack(np.sin(th[0:1]))


    def get_method(self,enum):
        return self.llist_E_DTLZ[enum]


    def param_F(self):
        dict_F = {
                    self.get_method(0) : self.F1,
                    self.get_method(1) : self.F2,
                    self.get_method(2) : self.F3,
                    self.get_method(3) : self.Fm
                  }
        return dict_F


    def calc_F_M(self,Fi,M):
        if Fi == 1:
            return self.get_method(0)
        elif Fi == 2 and M > 2:
            return self.get_method(1)
        elif Fi >= 3 and Fi <= M-1 and M > 3:
            return self.get_method(2)
        elif Fi == M:
            return self.get_method(3)


    def calc_TH(self,X,Gxm,M):
        return [X[:,Xi:Xi+1]*np.pi/2 if Xi == 0 else (np.pi/(4*(1+Gxm))*(1+2*Gxm*X[:,Xi:Xi+1]))  for Xi in range(0,M-1)]


    def calc_f(self,X,G):
        vet_F_M = [self.calc_F_M(F,self.get_M()) for F, i in enumerate(range(0,self.get_M()), start = 1)]
        return np.column_stack(list(map(lambda Key: self.param_F()[Key](self.get_M(),self.calc_TH(X,G,self.get_M()),G),vet_F_M)))


    def calc_g(self,X):
        return np.sum((X[:,self.get_M()-1:]-0.5)**2, axis = 1).reshape(X.shape[0],1)
    
    
    def set_Point_in_G(self,X):
       self._point_in_g = X
    

    def get_Point_in_G(self):
       return self._point_in_g


    def POFsamples(self):
        X = self.get_Points()
        X[:,self.get_M()-1:self.get_N()]=0.5
        self.set_Point_in_G(X)
        G = self.calc_g(X)
        F = self.eval_cons(self.calc_f(self.get_Point_in_G(),G))
        return F


    def evaluation(self,x,n_ieq):
        G=self.calc_g(x)
        F=self.calc_f(x,G)
        result =  {"F" : F}
        if n_ieq != 0:
            cons = self.constraits(F,1.25)
            const  = cons.reshape(cons.shape[0],1)
            result["G"] = const
            result["feasible"] = np.any((result["G"] <-0.00000000001)  | (result["G"] > 0.00000000001) )
        return result




from MoeaBench.base_moea import BaseMoea
import random
from deap import base, creator, tools
import array
import numpy as np


@mb.moeas.register_moea()
class NSGA2deap(BaseMoea):

  toolbox = base.Toolbox()
  result_evaluate = None

  def __init__(self,problem=None,population = 160, generations = 300, seed = 1):
    super().__init__(problem,population,generations,seed)  
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,) * self.get_M())
    creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)   
    NSGA2deap.toolbox.register("attr_float", self.uniform, 0, 1, self.get_N())
    NSGA2deap.toolbox.register("individual", tools.initIterate, creator.Individual, NSGA2deap.toolbox.attr_float)
    NSGA2deap.toolbox.register("population", tools.initRepeat, list, NSGA2deap.toolbox.individual)
    NSGA2deap.toolbox.register("evaluate",self.evaluate)
    self.evalue = NSGA2deap.toolbox.evaluate
    random.seed(1)
    NSGA2deap.toolbox.decorate("evaluate", tools.DeltaPenality(self.feasible,1000))
    NSGA2deap.toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=0, up=1, eta=20)
    NSGA2deap.toolbox.register("mutate", tools.mutPolynomialBounded, low=0, up=1, eta=20, indpb=1/self.get_N())
    NSGA2deap.toolbox.register("select", tools.selNSGA2)


  def uniform(self,low, up, size=None):
    try:
      return [random.uniform(a,b) for a,b in zip(low,up)]
    except TypeError as e:
      return [random.uniform(a,b) for a,b in zip([low]*size,[up]*size)]


  def evaluate(self,X):
    NSGA2deap.result_evaluate = self.evaluation_benchmark(X)
    return NSGA2deap.result_evaluate['F'][0]


  def feasible(self,X):
    self.evaluate(X)
    if 'G' in NSGA2deap.result_evaluate:
      if NSGA2deap.result_evaluate["feasible"]:
       return True
    return False
  

  def evaluation(self):
    pop = NSGA2deap.toolbox.population(n=self.get_population())
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = NSGA2deap.toolbox.map(NSGA2deap.toolbox.evaluate, invalid_ind)
    F_gen_all=[]
    X_gen_all=[]
    hist_F_non_dominate=[]
    hist_X_non_dominate=[]
    hist_F_dominate=[]
    hist_X_dominate=[]
    for ind, fit in zip(invalid_ind, fitnesses):
      ind.fitness.values = fit
    F_gen_all.append(np.column_stack([np.array([ind.fitness.values for ind in pop ])]))
    X_gen_all.append(np.column_stack([np.array([np.array(ind) for ind in pop ])]))
    pop = NSGA2deap.toolbox.select(pop, len(pop))
    for gen in range(1, self.get_generations()):
      offspring = tools.selTournamentDCD(pop, len(pop))
      offspring = [NSGA2deap.toolbox.clone(ind) for ind in offspring]
      for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
        if random.random() <= 0.9:
          NSGA2deap.toolbox.mate(ind1, ind2)
        NSGA2deap.toolbox.mutate(ind1)
        NSGA2deap.toolbox.mutate(ind2)
        del ind1.fitness.values, ind2.fitness.values
      invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
      fitnesses = NSGA2deap.toolbox.map(NSGA2deap.toolbox.evaluate, invalid_ind)
      for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
      pop = NSGA2deap.toolbox.select(pop + offspring, len(pop))
      F_gen = np.column_stack([np.array([ind.fitness.values for ind in pop ])])
      F_gen_all.append(F_gen)
      X_gen = np.column_stack([np.array([np.array(ind) for ind in pop ])])
      X_gen_all.append(X_gen)

      non_dominate = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
      
      F_non_dominate = np.column_stack( [np.array(  [ind.fitness.values for ind in non_dominate ])]    )
      hist_F_non_dominate.append(F_non_dominate)
    
      X_non_dominate = np.column_stack(  [np.array(  [ind for ind in non_dominate ]) ])
      hist_X_non_dominate.append(X_non_dominate)


      dominate = [ind for ind in pop if ind not in non_dominate]
      
      F_dominate = [ind.fitness.values for ind in dominate]
      F_dominate = np.array(F_dominate) if len(F_dominate) == 0 else np.column_stack(  [np.array(  F_dominate)])
      hist_F_dominate.append(F_dominate)


      X_dominate = [ind for ind in dominate]
      X_dominate = np.array(X_dominate) if len(X_dominate) == 0 else np.column_stack( [np.array( X_dominate)])
      hist_X_dominate.append(X_dominate)
       

    F = np.column_stack([np.array([ind.fitness.values for ind in pop ])])
    return F_gen_all,X_gen_all,F,hist_F_non_dominate,hist_X_non_dominate,hist_F_dominate,hist_X_dominate


exp5 = mb.experiment()
exp5.benchmark= mb.benchmarks.my_new_benchmark()
exp5.benchmark.M=5
exp5.moea = mb.moeas.my_new_moea()
exp5.moea.population = 100
exp5.moea.generations = 200

exp5.run(repeat = 10)

arr = exp5.dominated.objectives(generation = 5)
print(arr.shape)

arr = exp5.front(generation = 10)
print(arr.shape)


arr = exp5.objectives(generation = 10)
print(arr.shape)



#opt_front = exp5.optimal.front()
#print(opt_front)


#opt_set = exp5.optimal.set()
#print(opt_set)



#exp5.run()
#hv = exp5.hypervolume(generations = [150, 155])
#print(hv)

#exp5.benchmark.M = 4
#exp5.moea.generations = 400
#exp5.run()
#hv = exp5.hypervolume(generations = [395,399])
#print(hv)


#exp5.benchmark.M = 5
#exp5.moea.generations = 500
#exp5.population = 400

#exp5.run()
#hv = exp5.hypervolume(generations = [495,499])
#print(hv)

#moeabench.plot_hypervolume(exp5.result)
#exp5.save("gavan")


#exp5.moea.generations = 400
#exp5.moea.population = 260

#exp5.run()


#hv = exp5.hypervolume()
#print(hv)







