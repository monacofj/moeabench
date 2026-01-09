class front:

    def __init__(self, cls_result_front, result, rounds):
        self.cls_result_front = cls_result_front()
        self.result = result
        self.rounds = rounds


    def __call__(self, generation = None):
        try:
            return self.cls_result_front.IPL_front(self.result, generation)
        except Exception as e:
            print(e)
    

    def round(self, index):
        return self.rounds[index].front