from dash import Dash, html, dcc, Input, Output
import threading
from google.colab.output import eval_js
from .plot_3D import plot_3D
from .coordnates import coordnates
import plotly.graph_objects as go
import socket
import threading



class dashboard:

 def __init__(self,rank,idx_rank,variables):
    self.rank = rank
    self.idx_rank = idx_rank
    self.plot = plot_3D([0,1,2])
    self.coordenate = coordnates(variables)
    self.app = Dash(__name__)
    self.fig = self.plot.configure(rank,idx_rank)
    self.register_callback()


 def get_free_port(self):
        sock = socket.socket()
        sock.bind(("0.0.0.0", 0))
        port = sock.getsockname()[1]
        sock.close()
        return port
 

 def execute(self, rank_version = 1):
  port  = self.get_free_port()
  thread = threading.Thread(target=self.run, args =(port,),daemon=True)
  thread.start()
  url = eval_js(f"google.colab.kernel.proxyPort({port})")
  print(url)


 def build(self):
    self.app.layout = html.Div(
    [
        dcc.Store(
           id='rank_version',
           data = 0
        ),
        dcc.Graph(
            id ='paret',
            figure = self.fig
        ),
        dcc.Graph(
            id ='coordenate',
            figure = go.Figure()
        )
    ], style={
        'display': 'flex',
        'flexDirection': 'column',
        'alignItems': 'center'
    }
     )


 def run(self,port):
    self.app.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        use_reloader=False
    )


 def register_callback(self):

  self.build()

  @self.app.callback(Output('paret','figure'),
              Input('rank_version','data'),
            prevent_initial_call=True)

  def update_pareto(_):
   plot = plot_3D(self.exp,[0,1,2])

   return plot.configure(self.rank,self.idx_rank)



  @self.app.callback(Output('coordenate','figure'),
              Input('paret','clickData'),
            prevent_initial_call=True)

  def click_ponto(clickData):

   if clickData is None:
    return go.Figure()

   return self.coordenate.configure(clickData['points'][0]['customdata'])