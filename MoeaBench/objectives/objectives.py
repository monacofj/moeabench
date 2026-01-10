# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

class objectives:
    """
        - **array with objectives in generations:**
        Click on the links for more
        ...
                - **Informations:**
                      - sinxtase:
                      experiment.objective(args)  
                      - [objective](https://moeabench-rgb.github.io/MoeaBench/analysis/objectives/data/objective/) information about the method, examples and more...   
                      - [Exception](https://moeabench-rgb.github.io/MoeaBench/analysis/metrics/data/exceptions/) information on possible error types

        """
    def __init__(self, cls_result_obj, result, rounds):
        self.cls_result_obj = cls_result_obj()
        self.result = result
        self.rounds = rounds


    def __call__(self, generation = None):
        try:
            return self.cls_result_obj.IPL_objectives(self.result, generation)
        except Exception as e:
            print(e)
    

    def round(self, index):
        return self.rounds[index].objectives

