from .d_objectives import d_objectives
from .d_variables import d_variables

class dominated:

    def __init__(self, cls_result_dominated, result, rounds):
        self.cls_result_dominated = cls_result_dominated() 
        self.result = result
        self.rounds = rounds


    @property 
    def objectives(self, generation = None):
        try:
            obj = d_objectives(self.cls_result_dominated, self.result, self.rounds)
            return obj
        except Exception as e:
            print(e)


    @property 
    def variables(self, generation = None):
        try:
            var = d_variables(self.cls_result_dominated, self.result, self.rounds)
            return var
        except Exception as e:
            print(e)