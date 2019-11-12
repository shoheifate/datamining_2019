import numpy as np

def load_nonlinear_example1():
    """
    >>> X,Y = load_nonlinear_example1()
    >>> print(X[0])
    [1. 0.]
    """
    X = np.array([[1,0.0],[1,2.0],[1,3.9],[1,4.0]])
    Y = np.array([4.0,0.0,3.0,2.0])
    return X,Y

def polynomia2_features(input):
    poly2 = input[:,1:]**2
    return np.c_[input,poly2]

def polynomial3_features(input):
    poly2 = input[:,1:] ** 2
    poly3 = input[:,1:] ** 3
    return np.c_[input,poly2,poly3]