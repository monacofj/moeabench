from .IPL_MoeaBench import IPL_MoeaBench 
import numpy as np


class result_population(IPL_MoeaBench):

    @staticmethod
    def allowed_gen(generations):
        if generations is not None and not isinstance(generations, int):
            raise TypeError("Only variables of type int are allowed in 'generation'")    
         

    @staticmethod
    def allowed_gen_max(maximum, N):
        if N is not None and not 0 <= N <= maximum:
            raise TypeError(f"generations = {N} not be allowed. It must be between 0 and {maximum}" )
      

    def gen_data(self, gen_all, generations = None):
        return gen_all[-1] if generations is None else gen_all[generations]


    def DATA(self, gen_f_max, generation):
        result_population.allowed_gen(generation)
        result_population.allowed_gen_max(len(gen_f_max),generation)        
        return self.gen_data(gen_f_max,generation)