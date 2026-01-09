from MoeaBench.base_moea import BaseMoea
from MoeaBench.integration_moea import integration_moea
import random
from deap import base, creator, tools, algorithms
import array
import numpy as np


class my_NSGA2deap(integration_moea):
               
        def __init__(self,population = 160, generations = 300, seed = 1):
          super().__init__(NSGA2deap,population,generations,seed)


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





