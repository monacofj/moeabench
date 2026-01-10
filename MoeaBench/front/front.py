# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

class front:

    def __init__(self, cls_result_front, result, rounds):
        self.cls_result_front = cls_result_front()
        self.result = result
        self.rounds = rounds


    def __call__(self, generation = None):
        try:
            return self.cls_result_front.IPL_front(self.result, generation)
        except Exception as e:
            print(e)
    

    def round(self, index):
        return self.rounds[index].front