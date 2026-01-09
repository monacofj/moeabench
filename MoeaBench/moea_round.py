from .moea_dm_round import moea_dm_round

class moea_round:

    def __init__(self, obj, name):
        self._dominated = moea_dm_round(obj.get_F_gen_dominate()[-1], obj.get_X_gen_dominate()[-1])
        self._objectives = obj.get_F_GEN()[-1]
        self._variables = obj.get_X_GEN()[-1]
        self._front = obj.get_F_gen_non_dominate()[-1]
        self._set = obj.get_X_gen_non_dominate()[-1]
        self._name = name


    @property
    def name(self):
        return self._name
    

    @property
    def objectives(self):
        return self._objectives
    

    @property
    def variables(self):
        return self._variables
    

    @property
    def front(self):
        return self._front
    

    @property
    def set(self):
        return self._set
    

    @property
    def dominated(self):
        return self._dominated


   