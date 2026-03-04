# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import plotly.graph_objects as go

# from .analyse import analyse # Legacy removed

class ParallelCoordinate:

    def __init__(self, markers=None, label=None, metric=None):
        """
        markers: tuple/list (x_values, y_values)
        label: list of labels for traces
        metric: tuple (y_label, x_label, title)
        """
        self.markers = markers
        self.label = label
        self.metric = metric
        self.figure = None
     
    def show(self):
         self.configure()
         if self.figure:
             self.figure.show()

    def configure(self):
             title = None
             try:
                  title = self.metric[2]
             except Exception as e:
                  title = self.metric[0] if self.metric else "Chart"
                  
             x_label = self.metric[1] if self.metric and len(self.metric)>1 else "X"
             y_label = self.metric[0] if self.metric and len(self.metric)>0 else "Y"

             self.figure=go.Figure()
             
             if not self.markers: return

             # markers[0] is list of X arrays (one for each trace)
             # markers[1] is list of Y arrays
             xs = self.markers[0]
             ys = self.markers[1]
             
             for x_data, y_data, lbl in zip(xs, ys, self.label):
                 x_arr = np.array(x_data)
                 y_arr = np.array(y_data)
                 
                 self.figure.add_trace(go.Scatter(
                     x = x_arr, y = y_arr,
                     mode='lines+markers',
                     marker=dict(size=3),
                     name=f'{lbl}',
                     showlegend=True,
                     hovertemplate = (f"{lbl}<br>"
                                f"{x_label}: %{{x}}<br>"
                                f"{y_label}: %{{y}}<br><extra></extra>"),
                                
                                ))
                                
             self.figure.update_layout(       
                 xaxis=dict(title=x_label, showgrid=True, gridcolor="#C3BDBD"),
                 yaxis=dict(title=y_label, showgrid=True, gridcolor="#C3BDBD"),
                 margin=dict(l=70,r=150,b=80,t=140),
                 plot_bgcolor="#FAFAFA",
                 paper_bgcolor="white",
                 width=800,
                 height=700,
                 title=dict(
                     text=f'2D Chart for {title}',
                     x=0.5,
                     xanchor='center',
                     y=0.9,
                     yanchor='top',
                     pad=dict(t = 10, b = 140),
                     font=dict(size=16, 
                     weight='bold')),
                     legend=dict(x=1.05,
                                 y=0.5,
                                 xanchor='left',
                                 yanchor='middle',
                                 font=dict(size=11, weight='bold')))
      


                
                
        

                