from collections import defaultdict
import pandas as pd

class Logger:
    def __init__(self):
        self.step = 0
        self.data = defaultdict(dict)
        
    def set_step(self, step):
        self.step = step
        
    def _scalar_name(self, scalar_name):
        return f"{scalar_name}"
    
    def add_scalar(self, scalar_name, scalar):
        self.data[self._scalar_name(scalar_name)][self.step] = scalar
        
    def to_pandas(self):
        series = [pd.Series(name=scalar_name, data=value_by_step.values(), index=value_by_step.keys()) 
                  for scalar_name, value_by_step in self.data.items()]
        return pd.concat(series, axis=1)