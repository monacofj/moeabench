# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from .allowed_stats import allowed_stats
import plotly.express as px


class paretorank_instance(allowed_stats):

    def __init__(self, experiment):
        self.experiment = experiment
        self.ranking = []


    def allowed(self, exp):
        if not hasattr(exp,'result'):
            raise ValueError("only experiment data types are allowed.")    


    def __call__(self):
        #try:
            self.allowed(self.experiment)
            rank = [i for i in self.experiment.rounds]
            self.ranking = sorted(rank, key = lambda pop: pop.front.shape[0], reverse = True)
       # except Exception as e:
           # print(e)
    

    def rank(self):
        return [round.name for round in self.ranking]
    

    def plot(self):
        fig = px.bar(
            y = [nd.front.shape[0] for nd in self.ranking],
            x = [round.name for round in self.ranking],
            labels = {'y' : "nom dominated", 'x' : 'ranking'},
            title = "Rank"
        )
        fig.show()


