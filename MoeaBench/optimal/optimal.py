class optimal:

    def __init__(self, benchmark):
        self.benchmark = benchmark


    def front(self):
        return [pof.get_arr_DATA() for pof_benchmark in self.benchmark.pof.get_CACHE().get_elements() for pof in pof_benchmark if hasattr(pof,'get_arr_DATA')][0]
    

    def set(self):
        return self.benchmark.benchmark.get_Point_in_G()