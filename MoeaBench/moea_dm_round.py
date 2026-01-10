# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

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
    
