import numpy as np


def predict(x, w, b):

    # single predict using linear regression
    # Args:
    #   x (ndarray): Shape (n,) example with multiple features
    #   w (ndarray): Shape (n,) model parameters
    #   b (scalar):             model parameter

    # Returns:
    #   p (scalar):  prediction

    p = np.dot(x, w) + b
    return p
