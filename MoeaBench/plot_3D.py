# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import ipywidgets as widgets
from IPython.display import display
import plotly.graph_objects as go
import numpy as np
from .analyse_pareto import analyse_pareto
try:
    import google.colab
    from google.colab import output
    output.enable_custom_widget_manager()
except ImportError:
    pass


class plot_3D(analyse_pareto):
    
     def __init__(self, benk, vet_pt, axis, type = 'pareto-optimal front', mode='interactive', axis_label='Objective'):
         self.vet_pts=vet_pt
         self.experiments=benk
         self.axis = axis
         self.type=type
         self.mode=mode
         self.axis_label=axis_label
           

     def configure(self):
         if self.mode == 'static':
             self.configure_static()
         else:
             self.configure_interactive()

     def configure_static(self):
         import matplotlib.pyplot as plt
         fig = plt.figure(figsize=(10, 8))
         ax = fig.add_subplot(111, projection='3d')
         
         for i in range(0, len(self.vet_pts)):
            ax_data = self.vet_pts[i][:,self.axis[0]]
            ay_data = self.vet_pts[i][:,self.axis[1]]
            az_data = self.vet_pts[i][:,self.axis[2]]
            
            # Simple cleaning of NaNs if needed, similar to original logic
            msk = ~(np.isnan(ax_data) | np.isnan(ay_data) | np.isnan(az_data))
            if np.any(msk):
                ax.scatter(ax_data[msk], ay_data[msk], az_data[msk], label=f'{self.experiments[i]}')
        
         ax.set_xlabel(f"{self.axis_label} {self.axis[0]+1}")
         ax.set_ylabel(f"{self.axis_label} {self.axis[1]+1}")
         ax.set_zlabel(f"{self.axis_label} {self.axis[2]+1}")
         ax.set_title(f"3D Chart for {self.type}")
         ax.legend()
         plt.show()

     def configure_interactive(self):
         self.figure=go.Figure()
         for i in range(0, len(self.vet_pts)):
                ax = self.vet_pts[i][:,self.axis[0]]
                ay = self.vet_pts[i][:,self.axis[1]]
                az = self.vet_pts[i][:,self.axis[2]]
                msk = ~(np.isnan(ax) | np.isnan(ay) | np.isnan(az))
                if np.any(msk):
                 self.figure.add_trace(go.Scatter3d(
                 x=ax, y=ay, z=az,
                 mode='markers',
                 marker=dict(size=3),  
                 name=f'{self.experiments[i]}',                       
                 showlegend=True,
                 hovertemplate = (f"{self.experiments[i]}<br>"
                                  f"{self.axis_label} {self.axis[0]+1}: %{{x}}<br>"
                                  f"{self.axis_label} {self.axis[1]+1}: %{{y}}<br>"
                                  f"{self.axis_label} {self.axis[2]+1}: %{{z}}<br><extra></extra>"),
                 ))   
       
      
         self.figure.update_layout(
                scene = dict(
                    xaxis=dict(title=f"{self.axis_label} {self.axis[0]+1}", showbackground=True, backgroundcolor="aliceblue", showgrid=True, gridcolor="#C3BDBD"),
                    yaxis=dict(title=f"{self.axis_label} {self.axis[1]+1}", showbackground=True, backgroundcolor="aliceblue", showgrid=True, gridcolor="#C3BDBD"),
                    zaxis=dict(title=f"{self.axis_label} {self.axis[2]+1}", showbackground=True, backgroundcolor="aliceblue", showgrid=True, gridcolor="#C3BDBD"),
                    aspectmode='manual',
                    aspectratio=dict(x=1,y=1,z=1)
                 ),
                 
                 width=900,
                 height=800,
                 margin=dict(l=0,r=0,b=0,t=0),
                 title=dict(
                     text=f'3D Chart for {self.type}',
                     x=0.5,
                     xanchor='center',
                     y=0.9,
                     yanchor='bottom',
                     pad=dict(t=0),
                     font=dict(size=16,weight='bold')
                 ),
                 legend=dict(
                     x=1,
                     y=0.5,
                     xanchor='left',
                     yanchor='middle'
               )
            )
         self.PLT()


     