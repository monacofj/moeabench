from .I_DPF import I_DPF
from .E_DPF import E_DPF
import numpy as np
from .NU_chaos import NU_chaos
from .init_benchmark import InitBenchmark


class H_DPF(NU_chaos,I_DPF,InitBenchmark):

     def __init__(self,metodhs,methods_R1,methods_R2,**kwargs):
        list_E_DPF = list(E_DPF)
        metodhs = {x-1 for x in metodhs}
        self.__arr_ENUM = [list_E_DPF[iIE_DPF] for iIE_DPF in range(0,len(E_DPF)) if  iIE_DPF in metodhs]
        methods_R1 = {x-1 for x in methods_R1}
        self.__arr_ENUM_R1 = [list_E_DPF[iIE_DPF] for iIE_DPF in range(0,len(E_DPF)) if  iIE_DPF in methods_R1]
        methods_R2 = {x-1 for x in methods_R2}
        self.__arr_ENUM_R2 = [list_E_DPF[iIE_DPF] for iIE_DPF in range(0,len(E_DPF)) if  iIE_DPF in methods_R2]
        super().__init__(**kwargs)


     def get_method(self,enum):
        return self.__arr_ENUM[enum]
     

     def get_method_R1(self,enum):
        return self.__arr_ENUM_R1[enum]
     

     def get_method_R2(self,enum):
        return self.__arr_ENUM_R2[enum]

            
     def calc_F_D(self,Fi,D):
          if Fi == 1:
               return self.get_method(0)
          elif Fi >= 2 and Fi <= D-2:
               return self.get_method(1)
          elif Fi > 1 and Fi == D-1:
               return self.get_method(2)
          elif Fi == D:
               return self.get_method(3)
          

     def calc_F_C(self,Fi,U):
          if Fi < U:
               return E_DPF.FD1
          elif Fi == U:
               return E_DPF.FM  


     def param_F(self):
        dict_PD = {
                    self.get_method(0)   : self.Y1,
                    self.get_method(1)   : self.Y2,
                    self.get_method(2)   : self.Yd1,
                    self.get_method(3)   : self.Yd
                  }
        return dict_PD
         
          
     def calc_F_PD(self,X,chaos,Yd1,vet_chaos):
         redundat = [] 
         for row in range(0,vet_chaos.shape[0]):
            for col_next, col in enumerate(range(0,len(vet_chaos[row])), start = 1):
               redundat.append(float(chaos[col](row,col,col_next)))
         return np.concatenate((Yd1,np.array(redundat)
                                .reshape(X.shape[0],vet_chaos.shape[1])), axis = 1)
                

     def param_CHAOS(self):
        dict_CHAOS = {
           E_DPF.FD  : self.FD,
           E_DPF.FD1 : self.FD1,
           E_DPF.FM  : self.FM
        }
        return dict_CHAOS
     

     def show_in(self,constraits):
         return  {f'IN POF' : constraits}
       

     def calc_gijx(self,Fijx):
         return 0
     

     def B1(self):
        raise NotImplementedError("Not implemented")
     

     def B2(self):
        raise NotImplementedError("Not implemented")
     

     def Bmd1(self):
        raise NotImplementedError("Not implemented")
     

     def FD1(self,F,vet_chaos):
        raise NotImplementedError("Not implemented")
    

     def FM(self,F,vet_chaos):
        raise NotImplementedError("Not implemented")
     

     def Yd1 (self):
        raise NotImplementedError("Not implemented")
     

     def FD(self,F,vet_chao):
        raise NotImplementedError("Not implemented")
     

     def calc_F_P(self):
        raise NotImplementedError("Not implemented")
    

       

     