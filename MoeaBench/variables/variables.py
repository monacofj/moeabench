# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

class variables:
    """
        - **array with decision variables in generations:**
        Click on the links for more
        ...
                - **Informations:**
                      - sinxtase:
                      experiment.variable(args)  
                      - [variable](https://moeabench-rgb.github.io/MoeaBench/analysis/variables/data/variable/) information about the method, examples and more...   
                      - [Exception](https://moeabench-rgb.github.io/MoeaBench/analysis/metrics/data/exceptions/) information on possible error types

     """
    def __init__(self, cls_result_var, result, rounds):
        self.cls_result_var = cls_result_var()
        self.result = result
        self.rounds = rounds


    def __call__(self, generation = None):
        try:
            return self.cls_result_var.IPL_variables(self.result, generation)
        except Exception as e:
            print(e)
    

    def round(self, index):
        return self.rounds[index].variables

