from scipy.stats import ks_2samp
from .allowed_stats import allowed_stats
import plotly.express as px


class kstest_instance(allowed_stats):
    
    def __init__(self, args):
        self.args = args
        self.statistic = []
        self.pvalue = []


    def __call__(self):     
        try:
            self.allowed(self.args)
            valid_values = [i.objectives() if hasattr(i,'result') else i for i in self.args]
            for obj in range(0, valid_values[0].shape[1]):
                values = [  exp[:,obj-1].astype(float)  for exp in valid_values  ]
                stat, value = ks_2samp(values[0],values[1])
                self.statistic.append(float(stat))
                self.pvalue.append(float(value))
        except Exception as e:
            print(e)  


    def plot(self):
        fig = px.bar(
            y = self.statistic,
            x = self.pvalue,
            labels = {'y': "statistic",'x' : 'pvalue'},
            title = "kstest"
        )
        fig.show()
          
        
