import numpy as np


class LinearRegression:
    """
    >>> import regression
    >>> model = regression.LinearRegression()
    >>> model.x
    >>> #nothing
    """
    x = None
    theta = None
    y = None

    def fit(self,x,y):
        """
        >>> import regression
        >>> import importlib
        >>> import datasets
        >>> X,Y = datasets.load_linear_example1()
        >>> importlib.reload(regression)
        <module 'regression' from '/Users/e175752/datamining/regression-test/regression.py'>
        >>> model = regression.LinearRegression()
        >>> model.fit(X,Y)
        >>> print(model.theta)
        [5.30412371 0.49484536]
        """
        temp = np.linalg.inv(np.dot(x.T,x))
        self.theta = np.dot(np.dot(temp,x.T),y)
    def predict(self,x):
        """
        >>> import importlib
        >>> import datasets
        >>> import regression
        >>> X,Y = datasets.load_linear_example1()
        >>> importlib.reload(regression)
        <module 'regression' from '/Users/e175752/datamining/regression-test/regression.py'>
        >>> model = regression.LinearRegression()
        >>> model.fit(X,Y)
        >>> model.predict(X)
        array([ 7.28350515,  9.2628866 , 11.7371134 , 13.71649485])
        """
        return np.dot(x, self.theta)
    def score(self,x,y):
        """
        >>> import importlib
        >>> import datasets
        >>> import regression
        >>> X,Y = datasets.load_linear_example1()
        >>> importlib.reload(regression)
        <module 'regression' from '/Users/e175752/datamining/regression-test/regression.py'>
        >>> model = regression.LinearRegression()
        >>> model.fit(X,Y)
        >>> model.score(X,Y)
        1.2474226804123711
        """
        error = self.predict(x)-y
        return (error**2).sum()

class RidgeRegression(LinearRegression):
    alpha = None

    def __init__(self,alpha=0.1):
        self.alpha = alpha

    def fit(self,input,output):
        """
        >>> import importlib
        >>> import regression
        >>> import datasets
        >>> X,Y = datasets.load_nonlinear_example1()
        >>> ex_X = datasets.polynomial3_features(X)
        >>> importlib.reload(regression)
        <module 'regression' from '/Users/e175752/datamining/regression-test/regression.py'>
        >>> model = regression.RidgeRegression()
        >>> model.fit(ex_X, Y)
        >>> model.theta
        array([ 3.54259714, -1.24971967, -0.68925104,  0.23695052])
        """
        xTx = np.dot(input.T, input)
        I = np.eye(len(xTx))
        self.theta = np.dot(np.dot(np.linalg.inv(xTx + self.alpha * I),input.T), output)