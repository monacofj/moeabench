import numpy as np
from itertools import accumulate, repeat, cycle, islice

class NU_chaos:
   
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    
    def calc_NU_(self,Ci):
        return 3.8 * Ci * (1- Ci)
    

    def calc_NU(self,U,P,factor=1):
        VET = list(accumulate(repeat(None,factor), lambda acc, _:  self.calc_NU_(acc), initial=0.1))[1:]
        V_CYCLE = cycle(VET) 
        CHAOS= np.array(sorted(list(islice(V_CYCLE,P*U))))
        return CHAOS.reshape(P,U)
    

   
    