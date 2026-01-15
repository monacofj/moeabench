# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

M_register = {}

class my_new_moea:
    """
    Template for implementing a new MOEA.
    """
    
    def __init__(self, population=160, generations=300, seed=0):
        self.population = population   
        self.generations = generations      
        self.seed = seed
        self.problem = None

    def __call__(self, experiment, default=None, stop=None, seed=None):
        """Standard MoeaBench specialization entry point."""
        self.problem = experiment
        if seed is not None:
            self.seed = seed
        return self

    def evaluation(self):
        """
        Standard MoeaBench evaluation entry point.
        Should return (F_gens, X_gens, F_final, F_nd_history, X_nd_history, F_dom_history, X_dom_history).
        """
        # Placeholder for user implementation
        return [], [], None, [], [], [], []

def register_moea():
    def decorator(cls):
        try:
            name = cls.__name__
            M_register[name] = cls
        except Exception as e:
                print(e)
        return cls
    return decorator

def get_moea():
    return next(iter(M_register.values())) if len(M_register.values()) > 0 else None
