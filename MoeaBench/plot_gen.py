import numpy as np
import plotly.graph_objects as go
import numpy as np
from .analyse import analyse


class plot_gen(analyse):

    def __init__(self,markers = None, label = None, metric = None):
        self.markers=markers
        self.label=label
        self.metric=metric
     
    
    def configure(self):
             title = None
             try:
                  title = self.metric[2]
             except Exception as e:
                  title = self.metric[0]
             self.figure=go.Figure()
             self.figure.data=()
             for gen, metric,  lbl in zip( self.markers[0],  self.markers[1], self.label ):
                 gen=np.array(gen)
                 metric=np.array(metric)
                 self.figure.add_trace(go.Scatter(
                     x = gen, y = metric,
                     mode='lines+markers',
                     marker=dict(size=3),
                     name=f'{lbl}',
                     showlegend=True,
                     hovertemplate = (f"{lbl}<br>"
                                f"{self.metric[1]}: %{{x}}<br>"
                                f"{self.metric[0]}: %{{y}}<br><extra></extra>"),
                                
                                ))
                 self.figure.update_layout(       
                     xaxis=dict(title=self.metric[1], showgrid=True, gridcolor="#C3BDBD"),
                     yaxis=dict(title=self.metric[0], showgrid=True, gridcolor="#C3BDBD"),
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
             self.PLT()
      


                
                
        

                