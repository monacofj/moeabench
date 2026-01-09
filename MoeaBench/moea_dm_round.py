class moea_dm_round:

    def __init__(self, objectives, variables):
        self._objectives = objectives
        self._variables = variables


    @property
    def objectives(self):
        return self._objectives
    

    @property
    def variables(self):
        return self._variables
    
