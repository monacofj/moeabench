   
import numpy as np


class constraits_05:
    
    def constraits_05(self,f,parameter,f_c=[]):
        f_constraits=np.array(f)
        f_c = np.array(np.sum(f_constraits, axis = 1)-parameter).reshape(f_constraits.shape[0],1)
        return f_c
    
