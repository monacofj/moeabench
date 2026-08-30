from .dominance import dominance
from .dashboard import dashboard

class explorer:

  def __init__(self,objectives,variables):
    self.objectives = objectives
    self.variables = variables


  def execute(self):
    pareto = dominance()
    ranks = pareto.rank_pareto(self.objectives)
    idx_rank = list(dict.fromkeys(pareto.idx_rank(self.objectives,ranks)))
    idx_aux = pareto.associate_rank(ranks,idx_rank)
    db = dashboard(ranks,idx_aux,self.variables)
    db.execute()