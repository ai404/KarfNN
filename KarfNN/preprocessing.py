import numpy as np


class Scaler:
    """
        normalisation Class
        X <numpy.array>: input data matrix
        return <numpy.array>: normalized data
    """
    def __init__(self, X=None):
        if X is not None:
            self.fit(X)

    def fit(self, X):
        self.mx = X.max(axis=0)
        self.mi = X.min(axis=0)

    def fit_transform(self, X):
        self.fit(X)
        return (X - self.mi) / (self.mx - self.mi)

    def transform(self, X):
        return (X - self.mi) / (self.mx - self.mi)

    def reverse_transform(self, X):
        return X * (self.mx - self.mi) + self.mi


def polynomial_variables(X, poly_degree=2):
    """
        Generate polynomial Variables
        X <numpy.array>: original input data
        poly_degree <int>: polynomial degree
        return <numpy.array>: Calculated Polynomial Variables
    """
    n_cols = X.shape[1] if len(X.shape) > 1 else 1
    dot_product = X.copy().reshape(-1, 1) if n_cols == 1 else X.copy()

    a, b = 0, n_cols
    for k in range(2, poly_degree + 1):
        for j in range(n_cols):
            dot_product = np.column_stack([dot_product, dot_product[:, a + j:b] * dot_product[:, j].reshape(-1, 1)])
        a, b = b, dot_product.shape[1]

    return dot_product
