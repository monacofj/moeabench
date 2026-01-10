# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from pymoo.indicators.igd_plus import IGDPlus
import numpy as np


class GEN_igdplus:

    def __init__(self,hist_F,F,**kwargs):
        self.F=F
        self.hist_F=hist_F
        super().__init__(**kwargs)


    def evaluate(self):
        metric = IGDPlus(self.F, zero_to_one=True)
        return np.array([float(metric.do(_F)) for _F in self.hist_F])