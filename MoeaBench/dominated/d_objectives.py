class d_objectives:

    def __init__(self, cls_result_dominated, result, rounds):
        self.result_dominated = cls_result_dominated
        self.result = result
        self.rounds = rounds


    def __call__(self, generation = None):
       return self.result_dominated.IPL_dominated_objectives(self.result, generation)


    def round(self, index):
        return self.rounds[index].dominated.objectives 