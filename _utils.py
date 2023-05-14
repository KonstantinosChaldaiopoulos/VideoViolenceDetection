import numpy as np

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