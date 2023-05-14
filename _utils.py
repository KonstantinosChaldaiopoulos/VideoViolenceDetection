import numpy as np
import pandas as pd
from IPython.display import display

class Utils:

    def represent(array, metric):
        if metric == "means":
            return np.mean(array, axis=1)
        elif metric == "std":
            return np.std(array, axis=1)
        elif metric == "median":
            return np.median(array, axis=1)
        elif metric == "range":
            return np.ptp(array, axis=1)
        
    def visualize(tuple, keys):
        data = {}
        for i, element in enumerate(tuple):
            data[keys[i]] = list(element)
        df = pd.DataFrame(data)
        display(df)   