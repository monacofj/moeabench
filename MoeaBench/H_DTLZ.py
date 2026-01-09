from .I_DTLZ import I_DTLZ
from .E_DTLZ import E_DTLZ
from .init_benchmark import InitBenchmark


class H_DTLZ(I_DTLZ, InitBenchmark):

    def __init__(self,metodhs,**kwargs):
        list_E_DTLZ = list(E_DTLZ)
        metodhs = {x-1 for x in metodhs}
        self.__arr_ENUM = [list_E_DTLZ[iIE_DTLZ] for iIE_DTLZ in range(0,len(list_E_DTLZ)) if iIE_DTLZ in metodhs]
        super().__init__(**kwargs)

    
    def get_method(self,enum):
        return self.__arr_ENUM[enum]


    def calc_F_M(self,Fi,M):
        if Fi == 1:
            return self.get_method(0)
        elif Fi == 2 and M > 2:
            return self.get_method(1)
        elif Fi >= 3 and Fi <= M-1 and M > 3:
            return self.get_method(2)
        elif Fi == M:
            return self.get_method(3)
        

    def show_in(self,constraits):
        return  {F"IN POF": constraits[0]                           
                }if len(constraits[0]) > 0 else {F"NEAR POF": constraits[1]}
       

    def calc_gijx(self,Fijx):
        return 0
    

    def calc_g(self,X):
        return 0
    

    def F1(self):
        raise NotImplementedError("Not implemented")


    def F2(self):
        raise NotImplementedError("Not implemented")
    

    def F3(self):
        raise NotImplementedError("Not implemented")
    

    def Fm(self):
        raise NotImplementedError("Not implemented")
    
    
    def FiFj(self):
        raise NotImplementedError("Not implemented")
    

    def Gj(self):
        raise NotImplementedError("Not implemented")
    

    def Fm1(self):
        raise NotImplementedError("Not implemented")
    

    def Gm(self):
        raise NotImplementedError("Not implemented")
    

    def calc_MinFiFj(self):
        raise NotImplementedError("Not implemented")
    

    def calc_H(self):
        raise NotImplementedError("Not implemented")

    

