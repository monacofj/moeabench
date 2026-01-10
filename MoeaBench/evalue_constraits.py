# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np


class evalue_constraits:
    
    def eval_cons(self,f,type_cons=True):
        const_in=[]
        const_out=[]
        M_constraits = self.constraits_1(f,self.get_Pareto()) if type_cons == True else self.constraits_05(f,self.get_Pareto())
        for (fc,fo) in zip(M_constraits,f):
            const_in.append(fo) if float(fc) == 0 else const_out.append(fo)
        return np.array(const_in),np.array(const_out)