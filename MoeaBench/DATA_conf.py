from .I_DATA_conf import I_DATA_conf


class DATA_conf(I_DATA_conf):

    def set_DATA_MOEA(self,DATA_MOEA,problem):
        self.__DATA_MOEA=DATA_MOEA
        self.__problem=problem
        

    def set(self,description,generations,population,arr_DATA,F_GEN,X_GEN,F_gen_non_dominate,X_gen_non_dominate,F_gen_dominate,X_gen_dominate):
        self.__description=description
        self.__generations=generations
        self.__population=population
        self.__arr_DATA=arr_DATA
        self.__F_GEN=F_GEN
        self.__X_GEN=X_GEN
        self.__F_gen_non_dominate=F_gen_non_dominate
        self.__X_gen_non_dominate=X_gen_non_dominate
        self.__F_gen_dominate=F_gen_dominate
        self.__X_gen_dominate=X_gen_dominate


    def get_DATA_MOEA(self):
        return self.__DATA_MOEA
    
    
    def get_problem(self):
        return self.__problem
    

    def get_description(self):
        return self.__description


    def get_population(self):
        return self.__population


    def get_generations(self):
        return self.__generations
    

    def set_arr_DATA(self, value):
        self.__arr_DATA = value
    

    def get_arr_DATA(self):
        return self.__arr_DATA
    

    def get_F_GEN(self):
        return self.__F_GEN
    

    def get_F_gen_non_dominate(self):
        return self.__F_gen_non_dominate
    

    def get_F_gen_dominate(self):
        return self.__F_gen_dominate
    

    def get_X_gen_non_dominate(self):
        return self.__X_gen_non_dominate
    
    
    def get_X_GEN(self):
        return self.__X_GEN
    
    
    def get_X_gen_dominate(self):
        return self.__X_gen_dominate


    

    

    

    
    

    






