import numpy as np
from scipy.interpolate import griddata
import plotly.graph_objects as go
from .analyse_pareto import analyse_pareto
from scipy.ndimage import gaussian_filter


class plot_surface_3D(analyse_pareto):
    
    def __init__(self, experiments, vet_pt, vaxis,  type = 'pareto-optimal front'):
         self.color = {
             0: 'Viridis',
             1: 'Plasma',
             2: 'Cividis',
             3: 'Turbo',
             4: 'Inferno',
             5: 'Magma'

         }
         self.vet_pts=vet_pt
         self.experiments=experiments
         self.vaxis = vaxis
         self.type = type
         self.parameter=[{"F" :value,  "opacity" : 0.9, "showscale" : False, "showlegend" : True, "colorscale": self.color[index] } for index, value in enumerate(self.vet_pts, start = 0)]


    def axis(self,points,values,X,Y):
        try:
            z_cubic = griddata(points=points,values=values, xi = (X,Y), method='cubic')
            if not np.isnan(z_cubic).any():
                return z_cubic
        except Exception as e:
            pass
        try:
            z_nearest = griddata(points=points,values=values, xi = (X,Y), method='nearest')
            z_linear = griddata(points=points,values=values, xi = (X,Y), method='linear')
            return np.where(np.isnan(z_linear),z_nearest,z_linear)
        except Exception as e:
            pass
        try:
            return griddata(points=points,values=values, xi = (X,Y), method='nearest')
        except Exception as e:
            raise RuntimeError("No valid Z-axis value found") from e


    def DATA(self,exp,F=[],opacity=0.7,showscale=True, showlegend=True,colorscale=None,label=[],x_axis=[],y_axis=[],z_axis=[]):
        grid = 120
        xi = np.linspace(F[:,x_axis].min(),F[:,x_axis].max(),grid)
        yi = np.linspace(F[:,y_axis].min(),F[:,y_axis].max(),grid)
        X,Y = np.meshgrid(xi,yi)
        points=F[:,[x_axis,y_axis]]
        values=F[:,z_axis]
        Z = self.axis(points,values,X,Y)
        Z = gaussian_filter(Z,sigma=1.0)
        return go.Surface(
            x=X,y=Y,z=Z,
            opacity=opacity,
            showscale=showscale,
            showlegend=showlegend,
            colorscale=colorscale,
            name=label,
            hovertemplate = (f"{exp}<br>"
                                  f"{self.vaxis[0]+1}: %{{x}}<br>"
                                  f"{self.vaxis[1]+1}: %{{y}}<br>"
                                  f"{self.vaxis[2]+1}: %{{z}}<br><extra></extra>")
                )


    def configure(self):
     try:
        self.list_axis = np.array([[0,1,2] for i in range(0,len(self.experiments)+1)])
        self.figure=go.Figure()    
        surfaces = [self.DATA(
            exp=exp,
            F=pr['F'],
            opacity=pr['opacity'],
            showscale=pr['showscale'],
            showlegend=pr['showlegend'],
            colorscale=pr['colorscale'],
            label=exp,
            x_axis=ax[0],
            y_axis=ax[1],
            z_axis=ax[2])
            for ax, pr,exp in zip(self.list_axis,self.parameter,self.experiments)]
        self.figure.add_traces(surfaces)
        self.figure.update_layout(
                scene = dict(
                    xaxis=dict(title=self.vaxis[0]+1, showbackground=True, backgroundcolor="aliceblue", showgrid=True, gridcolor="#C3BDBD"),
                    yaxis=dict(title=self.vaxis[1]+1, showbackground=True, backgroundcolor="aliceblue", showgrid=True, gridcolor="#C3BDBD"),
                    zaxis=dict(title=self.vaxis[2]+1, showbackground=True, backgroundcolor="aliceblue", showgrid=True, gridcolor="#C3BDBD"),
                    aspectmode='manual',
                    aspectratio=dict(x=1,y=1,z=1)
                 ),
                 
                 width=900,
                 height=800,
                 margin=dict(l=0,r=0,b=0,t=0),
                 title=dict(
                     text=f'3D Surface Chart {self.type}',
                     x=0.5,
                     xanchor='center',
                     y=0.9,
                     yanchor='top',
                     pad=dict(t=0),
                     font=dict(size=16, weight='bold')
                 ),
                 legend=dict(
                     x=1,
                     y=0.5,
                     xanchor='right',
                     yanchor='middle'
               )
               )
        self.PLT()        
     except Exception as e:
          print(e)

    
   