from pymoo.indicators.gd import GD
import numpy as np


class GEN_gd:

    def __init__(self,hist_F,F,**kwargs):
        self.F=F
        self.hist_F=hist_F
        super().__init__(**kwargs)


    def evaluate(self):
        metric = GD(self.F, zero_to_one=True)
        return np.array([float(metric.do(_F)) for _F in self.hist_F])