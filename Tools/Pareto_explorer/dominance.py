import numpy as np

class dominance:

 def comparation(self,arr, target):
  cd1 = np.all (arr[target] <= arr , axis = 1)
  cd2 = np.any (arr[target] < arr, axis = 1)
  condition = cd1 & cd2
  return condition


 def dominate(self,condition,i):
  for t in range(0,len(condition)):
    if condition[t][i] == True:
      return True
  return False


 def analyse(self,arr):
  nd = []
  arr_unique, arr_repetead = np.unique(arr, axis = 0, return_index=True)
  unique = arr_unique[np.argsort(arr_repetead)]
  repetead = arr[np.setdiff1d(np.arange(len(arr)),arr_repetead)]
  condition = []
  for i in range(0,len(unique)):
    condition.append(self.comparation(unique,i))

  for g in range(0,len(unique)):
    flag = self.dominate(condition,g)
    if flag is False:
      nom_dominated = unique[g]
      nom_dominated_repeat = np.where(np.all( repetead  == nom_dominated, axis = 1))[0]
      nd.append(nom_dominated)
      if len(nom_dominated_repeat) > 0:
          for z in nom_dominated_repeat:
            nd.append(repetead[z])
  return np.array(nd)


 def rank_pareto(self,arr):
  ranks = []
  mask = np.ones(len(arr), dtype=bool)
  while len(arr[mask]) > 0:
   nd = self.analyse(arr[mask])
   ranks.append(nd)
   for x in nd:
    idx = np.where(np.all( np.array(arr) == x, axis = 1) & mask )[0]
    mask[idx[0]]=False
  return ranks


 def associate_rank(self,ranks,idx_rank):
  idx_aux = []
  b = 0
  for i in range(0,len(ranks)):
     idx = []
     for z in range(0,len(ranks[i])):
      idx.append(idx_rank[b])
      b += 1
     idx_aux.append(idx)
  return idx_aux


 def idx_rank(self,arr,rank):
  idx_aux = []
  m = np.array(arr)
  idx = []
  for b in rank:
      for i in b:
       found = np.where(np.all( m  == i, axis = 1))[0]
       for f in found:
        idx.append(int(f))
  return idx