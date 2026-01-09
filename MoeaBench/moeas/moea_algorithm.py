from MoeaBench import moeas


from typing import TYPE_CHECKING
if TYPE_CHECKING: from kernel_moea import NSGA_pymoo, SPEA_pymoo, UNSGA_pymoo, RVEA_pymoo, MOEAD_pymoo
if TYPE_CHECKING: from MoeaBench.CACHE import CACHE


class moea_algorithm:

    def __init__(self):
        self.__memory=moeas.CACHE()
    

    def MOEAD(self):
        return moeas.kernel_moea.MOEAD_pymoo, moeas.E_MOEA_algorithm.MOEAD_pymoo
               

    def NSGA3(self):
        return moeas.kernel_moea.NSGA_pymoo, moeas.E_MOEA_algorithm.NSGA_pymoo
    

    def SPEA2(self,):
        return moeas.kernel_moea.SPEA_pymoo, moeas.E_MOEA_algorithm.SPEA_pymoo
            

    def U_NSGA3(self):
        return moeas.kernel_moea.UNSGA_pymoo, moeas.E_MOEA_algorithm.UNSGA_pymoo,


    def RVEA(self):
        return moeas.kernel_moea.RVEA_pymoo, moeas.E_MOEA_algorithm.RVEA_pymoo
    

    def my_new_moea(self):
        return moeas.my_new_moea, moeas.my_new_moea.__name__


    def dict_data(self):
        return {moeas.E_MOEA.NSGA3 : self.NSGA3,
                moeas.E_MOEA.SPEA2 : self.SPEA2,
                moeas.E_MOEA.U_NSGA3 : self.U_NSGA3,
                moeas.E_MOEA.MOEAD : self.MOEAD,
                moeas.E_MOEA.RVEA: self.RVEA,
                moeas.E_MOEA.my_new_moea: self.my_new_moea
                }


    def get_CACHE(self):
        return self.__memory
    

    def get_MOEA(self,name):
        moea_list = [moea for moea in list(moeas.E_MOEA) if moea.name == name]
        return self.dict_data()[moea_list[0]]() if len(moea_list) > 0 else False

