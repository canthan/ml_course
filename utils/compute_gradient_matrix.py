import numpy as np


def compute_gradient_matrix(X, y, w, b):
    """
    Computes the gradient for linear regression

    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
    Returns
      dj_dw (ndarray (n,1)): The gradient of the cost w.r.t. the parameters w.
      dj_db (scalar):        The gradient of the cost w.r.t. the parameter b.

    """
    m, n = X.shape
    f_wb = X @ w + b
    e = f_wb - y
    dj_dw = (1/m) * (X.T @ e)
    dj_db = (1/m) * np.sum(e)

    return dj_db, dj_dw
