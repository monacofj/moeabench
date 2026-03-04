# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from pymoo.indicators.hv import Hypervolume
import numpy as np


class GEN_hypervolume:

    def __init__(self,hist_F,M,approx_ideal,approx_nadir,ref_point=1.1,**kwargs):
        self.hist_F=hist_F
        self.M=M
        self.approx_ideal=approx_ideal
        self.approx_nadir=approx_nadir
        self.ref_point_val = ref_point
        super().__init__(**kwargs)


    def evaluate(self):
        # We construct the reference point array based on the single scalar value provided
        # This assumes the reference point is symmetric in the normalized space
        ref_arr = np.array([self.ref_point_val for i in range(0,self.M)])
        
        metric = Hypervolume(ref_point=ref_arr,
                     norm_ref_point=False,
                     zero_to_one=True,
                     ideal=self.approx_ideal,
                     nadir=self.approx_nadir)
        return np.array([float(metric.do(_F)) for _F in self.hist_F])