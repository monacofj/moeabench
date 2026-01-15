# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from pymoo.indicators.hv import Hypervolume
import numpy as np


class GEN_hypervolume:

    def __init__(self,hist_F,M,approx_ideal,approx_nadir,**kwargs):
        self.hist_F=hist_F
        self.M=M
        self.approx_ideal=approx_ideal
        self.approx_nadir=approx_nadir
        super().__init__(**kwargs)


    def evaluate(self):
        metric = Hypervolume(ref_point= np.array([1.1 for i in range(0,self.M)]),
                     norm_ref_point=False,
                     zero_to_one=True,
                     ideal=self.approx_ideal,
                     nadir=self.approx_nadir)
        return np.array([float(metric.do(_F)) for _F in self.hist_F])