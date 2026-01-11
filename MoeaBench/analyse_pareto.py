# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from .analyse import analyse
import numpy as np


class analyse_pareto(analyse):
    
    @staticmethod
    def dict_data(idx):
        return {0: f"array {idx}"
        }


    @staticmethod
    def DATA(i):
        if hasattr(i,'result') and hasattr(i.result,'get_elements'):
            return [z.get_F_GEN()[-1] for b in i.result.get_elements() for z in b if hasattr(z,'get_F_GEN')][0]  
        else:
            return None


    @staticmethod    
    def allowed_DATA(i):
        if hasattr(i,'result') and hasattr(i.result,'get_elements'):
            return True
        elif isinstance(i,np.ndarray) and i.ndim == 2:
            return True
        else:
            return False
        

    @staticmethod
    def allowed_obj_equal(data, benk):   
        arr = [i.shape[1] for i in data]
        if len(set(arr)) > 1:
            objs = [f'{benk[idx]} = {i} objectives' for idx, i in enumerate(arr, start = 0)]  
            raise ValueError (f'{objs} must be equals')   
          

    @staticmethod
    def allowed_obj(objectives, data, benk):
        if not isinstance(objectives, (list)):
            raise TypeError("Only arrays are allowed in 'objectives'")
        if  0 < len(objectives) < 3:
            raise TypeError(f"objectives = {objectives} not be allowed. I is necessary to follow the format: objectives = [obj1, obj2, obj3] " )       
        analyse_pareto.allowed_obj_equal(data, benk)


    @staticmethod
    def extract_pareto_result(args):
        idx = [i for i in range(1,len(args)+1)]
        val = np.array(list(map(lambda key: analyse_pareto.allowed_DATA(key),[i for i in args])))
        data = []
        benk = []
       
        if len(np.where(val == False)[0]):
            raise TypeError(f'incorrect data format: {[args[i] for i in range(0,len(val)) if val[i] == False] [0]  }')
        
        it_exp = iter(idx)
        it_arr = iter(idx)
        for i in args:
            arr = analyse_pareto.DATA(i)
            # Use _name if present, otherwise fall back to class name + index
            name = getattr(i, '_name', None)
            
            if name is None:
                if not isinstance(i, np.ndarray):
                    name = f'{i.__class__.__name__} {next(it_exp)}'
                else:
                    name = analyse_pareto.dict_data(next(it_arr))[0]
            
            arr = arr if arr is not None else i
            data.append(arr)
            benk.append(name)
        return benk, data
       
       
         
    

    

    

    


