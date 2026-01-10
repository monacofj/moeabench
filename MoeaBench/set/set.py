# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

class set:

    def __init__(self, cls_result_set, result, rounds):
        self.cls_result_set = cls_result_set()
        self.result = result
        self.rounds = rounds


    def __call__(self, generation = None):
        try:
            return self.cls_result_set.IPL_set(self.result, generation)
        except Exception as e:
            print(e)
    

    def round(self, index):
        return self.rounds[index].set