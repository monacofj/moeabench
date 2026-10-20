import numpy as np
import plotly.graph_objects as go

class plot_3D:

     def __init__(self, axis):
         #self.experiments=benk
         self.axis = axis


     def configure(self, pts, idx_aux = 0):
         fig=go.Figure()
         for vet_pt in range(0,len(pts)):
                ax = pts[vet_pt][:,self.axis[0]]
                ay = pts[vet_pt][:,self.axis[1]]
                az = pts[vet_pt][:,self.axis[2]]

                msk = ~(np.isnan(ax) | np.isnan(ay) | np.isnan(az))
                if np.any(msk):


                 trace = go.Scatter3d(
                 x=ax, y=ay, z=az,
                 mode='markers',
                 customdata = idx_aux[vet_pt],
                 marker=dict(size = 6),
                 name=f'Rank {vet_pt + 1 }',
                 showlegend=True,
                 hovertemplate =

                                  (
                                  f"Rank {vet_pt+1}<br>"
                                  "vector objective: %{customdata}<br>"
                                  f"{self.axis[0]+1}: %{{x}}<br>"
                                  f"{self.axis[1]+1}: %{{y}}<br>"
                                  f"{self.axis[2]+1}: %{{z}}<br>"
                                  "<extra></extra>"),
                 )
                 fig.add_trace(trace)







         fig.update_layout(
                scene = dict(
                    xaxis=dict(title=self.axis[0]+1, showbackground=True, backgroundcolor="aliceblue", showgrid=True, gridcolor="#C3BDBD"),
                    yaxis=dict(title=self.axis[1]+1, showbackground=True, backgroundcolor="aliceblue", showgrid=True, gridcolor="#C3BDBD"),
                    zaxis=dict(title=self.axis[2]+1, showbackground=True, backgroundcolor="aliceblue", showgrid=True, gridcolor="#C3BDBD"),
                    aspectmode='manual',
                    aspectratio=dict(x=1,y=1,z=1)
                 ),

                 width = 1000,
                 height=700,
                 margin=dict(l=50,r=50,b=0,t=0),
                 title=dict(
                     text=f'Pareto-optimal front',
                     x=0.5,
                     xanchor='center',
                     y=0.9,
                     yanchor='bottom',
                     pad=dict(t=0),
                     font=dict(size=16,weight='bold')
                 ),
                 legend=dict(
                     x=0.85,
                     y=0.5,
                     xanchor='left',
                     yanchor='middle'
               )
            )
         return fig