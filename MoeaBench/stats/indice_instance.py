from scipy import stats
import numpy as np
import pandas as pd
from .allowed_stats import allowed_stats


class indice_instance(allowed_stats):

    def __init__(self, cls_result_population, experiment, generation):
        self.experiment = experiment 
        self.generation = generation
        self.result_population = cls_result_population()
        self.table = {
            "GEN" : [],
            "mean" : [],
            "variance" : [],
            "std_dev" : [],
            "skewness" : [],
            "kurtosis" : []
            }
       

    def allowed(self, exp):
        if not hasattr(exp,'result'):
            raise ValueError("only experiment data types are allowed.")   
        valid = [types for exp in self.experiment.result.get_elements() for types in exp if hasattr(types,'get_F_GEN')]      
        self.result_population.allowed_gen(self.generation)
        self.result_population.allowed_gen_max(len(valid[0].get_F_GEN()),self.generation)
        return valid


    def __call__(self):
        try:
            valid = self.allowed(self.experiment)
            self.generation = self.generation if self.generation > 0 else -1
            for arr in valid:            
                        self.table["GEN"].append(self.generation)
                        self.table["mean"].append(np.mean(arr.get_F_GEN()[self.generation]))
                        self.table["variance"].append(np.var(arr.get_F_GEN()[self.generation]))
                        self.table["std_dev"].append(np.std(arr.get_F_GEN()[self.generation]))
                        self.table["skewness"].append(stats.skew(arr.get_F_GEN()[self.generation])[0])
                        self.table["kurtosis"].append(stats.kurtosis(arr.get_F_GEN()[self.generation])[0])
            df = pd.DataFrame(self.table)
            df.index = df.index+1
            return df.to_string(index = False)
        except Exception as e:
            print(e)  
