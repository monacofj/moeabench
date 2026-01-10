# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from .I_MoeaBench import I_MoeaBench
import numpy as np
from .GEN_hypervolume import GEN_hypervolume
from .GEN_gd import GEN_gd
from .GEN_gdplus import GEN_gdplus
from .GEN_igd import GEN_igd
from .GEN_igdplus import GEN_igdplus


class IPL_MoeaBench(I_MoeaBench):

    def IPL_variables(self):
        raise NotImplementedError("Not implemented")
    

    def DATA(self):
        raise NotImplementedError("Not implemented")


    def axis(self):
        raise NotImplementedError("Not implemented")

    
    def IPL_objectives(self):
        raise NotImplementedError("Not implemented")
    

    def IPL_front(self):
        raise NotImplementedError("Not implemented")
    

    def IPL_set(self):
        raise NotImplementedError("Not implemented")
    
    
    def IPL_plot_3D(self):
        raise NotImplementedError("Not implemented")
    
    
    def PLT(self):
        raise NotImplementedError("Not implemented")  
            
    
    def configure(self):
        raise NotImplementedError("Not implemented")
    

    def  IPL_GD(self):
        raise NotImplementedError("Not implemented")
    
    
    def IPL_GDplus(self):
        raise NotImplementedError("Not implemented")
    
    
    def IPL_IGD(self):
        raise NotImplementedError("Not implemented")
    
    
    def IPL_IGDplus(self):
        raise NotImplementedError("Not implemented")
      
    
    def IPL_hypervolume(self):
        raise NotImplementedError("Not implemented")
    

    def allowed_gen(self):
        raise NotImplementedError("Not implemented")
    

    def IPL_plot_GD(self):
        raise NotImplementedError("Not implemented")
    
    
    def IPL_plot_GDplus(self):
        raise NotImplementedError("Not implemented")
    
    
    def IPL_plot_IGD(self):
        raise NotImplementedError("Not implemented")
    
    
    def IPL_plot_IGDplus(self):
        raise NotImplementedError("Not implemented")
    
    
    def IPL_plot_hypervolume(self):
        raise NotImplementedError("Not implemented")
    

    def IPL_loader(self):
        raise NotImplementedError("Not implemented")
    

    def IPL_save(self):
        raise NotImplementedError("Not implemented")
    
    
    def F(self):
        raise NotImplementedError("Not implemented")
    

    def X(self):
        raise NotImplementedError("Not implemented")
    

    def dict_data(self):
        raise NotImplementedError("Not implemented")
    

    def verify(self):
        raise NotImplementedError("Not implemented")
    

    def gen_data(self):
        raise NotImplementedError("Not implemented")
    
      
    def extract_pareto_result(self):
        raise NotImplementedError("Not implemented")
    

    def IPL_dominated_objectives(self):
        raise NotImplementedError("Not implemented")
    

    def IPL_dominated_variables(self):
        raise NotImplementedError("Not implemented")
    
   
    @staticmethod
    def normalize(ref, F):
            data = [args.result.get_elements() 
                   
            for args in ref 
            if isinstance(args, object) 
            and hasattr(args,'result') 
            and hasattr(args.result,'get_elements')]
        
            gen = [n_dom.get_F_gen_non_dominate()[-1] 
            for element in data 
            for exp in element 
            for n_dom in exp 
            if hasattr(n_dom,"get_F_gen_non_dominate")]
            
           
            valid = [[np.min(i, axis = 0),np.max(i, axis = 0)]  for i in gen if len(i) > 0]
            if len(valid) > 0:
                individual_min = []
                individual_max = []
                for i in valid:
                    individual_min.append(i[0])
                    individual_max.append(i[1])

                general_mim = np.vstack((individual_min))
                general_max = np.vstack((individual_max))
                return np.min(general_mim , axis = 0), np.max(general_max , axis = 0)
            elif len(valid) == 0:
                return np.min(F[0], axis = 0), np.max(F[0], axis = 0)
            
    
    @staticmethod
    def slicing_arr(slc,arr):
        return np.hstack([arr[:,i:j]  for i,j in slc])
    

    @staticmethod
    def set_hypervolume(F_GEN, F, min_non, max_non):
        return [GEN_hypervolume(fgen,f.shape[1],min_non,max_non) for fgen,f in zip(F_GEN,F)]
    
    
    @staticmethod
    def set_GD(F_GEN,F):
        return [GEN_gd(fgen,f) for fgen,f in zip(F_GEN,F)]
    

    @staticmethod
    def set_GDplus(F_GEN,F):
        return [GEN_gdplus(fgen,f) for fgen,f in zip(F_GEN,F)]

    
    @staticmethod
    def set_IGD(F_GEN,F):
        return [GEN_igd(fgen,f) for fgen,f in zip(F_GEN,F)]

    
    @staticmethod
    def set_IGD_plus(F_GEN,F):
        return [GEN_igdplus(fgen,f) for fgen,f in zip(F_GEN,F)]
    

    @staticmethod
    def allowed_obj(objective):
        if not isinstance(objective, (list)):
            raise TypeError("Only arrays are allowed in 'objectives'")
        objective_set = list({x for x in objective})
        if not len(objective_set) == len(objective):
            raise ValueError("There are repeated elements for objectives")


    @staticmethod
    def allowed_obj_equal(element, data, experiments, objectives, obj = ('get_M',)):
        list_valid = list(map(lambda o: o.get_M(), filter(lambda o: all(hasattr(o,m) for m in obj), element)))
        if not all(np.array_equal(data.get_M(),arr) for arr in list_valid):
            objs = [f'{experiments[idx]} = {i.get_M()} objectives' for idx, i in enumerate(element, start = 0)]
            raise ValueError (f'{objs} must be equals')   
        less = [i if i > element[0].get_elements()[0][1].get_M() else f'obj' for idx, i in enumerate(objectives, start = 0)  ]
        digit = [i for i in less if str(i).isdigit()]
        if digit:
            raise ValueError (f'Objective(s) {less} can´t be greather than {element[0].get_elements()[0][1].get_M()}')  
 

    def allowed_DATA(LIST, experiments):             
        for IDATA,exp in zip(LIST,experiments):
            print(exp.__class__.__name__,"  ", len(IDATA.get_arr_DATA()))

        INF = [f'{IDATA.get_description()}' for IDATA in LIST if np.isinf(IDATA).any()] 
        if len(INF) > 0:
            raise ValueError(f'There are matrices with invalid values: '+",".join(f'{i}' for i in INF))
        

    @staticmethod
    def allowed_gen(generations):
        if not isinstance(generations, (list)):
            raise TypeError("Only arrays are allowed in 'generations'")
        if not len(generations) == 2:
            raise TypeError(f"generations = {generations} not be allowed. I is necessary to follow the format: generations = [begin, end]" )
        if not generations[0] <= generations[1]:
            raise TypeError("the initial generation must be smaller than the final generation")
        

    @staticmethod
    def allowed_gen_max(maximum, N):
        if not N <= maximum:
            raise TypeError(f"generations = {N} not be allowed. It must be between 0 and {maximum}" )
      
    

       




