import plotly.graph_objects as go
import numpy as np
import ipywidgets as widgets
from IPython.display import display
from plotly.graph_objs import FigureWidget
import numpy as np

class coordnates:

 def __init__(self, variables):
          self.variables=variables
          

 def configure(self, idx = None):
  lim = [idx,self.variables.shape[0]]
  prev = np.sort([lim[0]-i  for i in range(0,11) if lim[0]-i >=0 and lim[0]-i != lim[0]])
  point_idx = np.array([idx])
  next = [lim[0]+i  for i in range(0,11) if lim[0]+i < lim[1]  and lim[0]+i != lim[0]]
  var_idx  = []
  if len(prev) > 0 and len(next) > 0:
    var_idx = np.concatenate((prev,point_idx,next))
  elif len(prev) == 0 and len(next) > 0:
    var_idx = np.concatenate((point_idx,next))
  elif len(prev) > 0 and len(next) == 0:
    var_idx = np.concatenate((prev,point_idx))

  if idx is not None:
    arr = [
    dict(
        label=f'X{b+1}',
        values=self.variables[var_idx,b],
        range=[0, 1]
    )
    for b in range(self.variables.shape[1])]
    colors = np.zeros(len(var_idx))
    pos = np.where(var_idx == idx)[0][0]
    colors[pos]=1
    fig = go.Figure()
    fig.add_trace(
     go.Parcoords(
        line = dict(

          color = colors,
          colorscale=[
        [0.0, 'gray'],
        [1.0, 'red']
    ]
        ),
        dimensions=arr
    ))
    fig.update_layout(

                     width = 1300,
                     height=500,
                     title=dict(
                     text=f'Decision variables for solution {idx}',
                     x=0.5,
                     xanchor='center',
                     y=0.9,
                     yanchor='bottom',
                     pad=dict(t=0),
                     font=dict(size=16,weight='bold')
                 ),
            margin=dict(l=30,r=30,b=0,t=180)
  )
  return fig