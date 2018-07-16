import os
import pandas as pd
import numpy as np

datasets = [
    "Iris",
    "brest_cancer"
]

datasets_path = os.path.dirname(__file__)


def load(name):
    """ Used to load available datasets """
    if name in datasets:

        return pd.read_csv(os.path.join(datasets_path, "%s.csv" % name))
    else:
        raise ValueError("Dataset not found!")
