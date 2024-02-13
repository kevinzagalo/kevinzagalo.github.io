"""
Packages:
    numpy as np
    matplotlib.pyplot as plt

Functions:
    plotXY
    add_bias
    grad_descent
    MSE
    fmin_bfgs
    plot_frontiere
    time
"""

print(__doc__)

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.metrics import mean_squared_error as MSE
from scipy.optimize import fmin_bfgs
from time import time



def plotXY(X, Y, legend=True):
    """
        Scatter points with a color for each class.
        Input:
            X and Y may be:
            - two numpy arrays with two columns; each array is the data matrix for a class (works only for
            two classes).
            - a (n,2)-numpy array with two columns (the data matrix) and the (n,)-vector of labels (works for many classes).
    """    
    if Y.ndim > 1:
        X1 = X
        X2 = Y
        print(X)
        print(Y)
        XX = np.concatenate((X, Y), axis=0)
        YY = np.concatenate((np.ones(X.shape[0]), -np.ones(Y.shape[0])))
    else:
        XX = X
        YY = Y
    for icl, cl in enumerate(np.unique(YY)):
        plt.scatter(XX[YY==cl, 0], XX[YY==cl, 1], label='Class {0:d}'.format(icl+1))
    plt.axis('equal')
    if legend:
        plt.legend()


def add_bias(XX):
    nn = XX.shape[0]
    return np.concatenate([np.ones((nn,1)), XX], axis=1)

def grad_descent(grad, w0, alpha=0.00008, max_iter=None, tol=1e-4, verbose=False, solver="auto"):
    step = 0
    err = tol+1
    
    # Transform w0 to the right dimension
    w0 = np.array(w0)
    if w0.ndim == 1:
        w0 = w0.reshape(-1, 1)
    elif w0.ndim == 0:
        w0 = w0.reshape(1, 1)
        
    if max_iter is None:
        max_iter = np.inf

    # Updating loop until convergence
    while step < max_iter and err > tol:
        w1 = w0 - alpha * grad(w0)
        step += 1
        err = np.sqrt(((w1-w0)**2).sum())
        w0 = w1


    if verbose:
        print('Converged in {} steps with precision {}'.format(step, err))
        print("The optimal parameter is: ", w1.T[0])

    return w1.T[0]


def plot_frontiere(clf, data=None, num=500, label=None):
    """
        Plot the frontiere f(x)=0 of the classifier clf within the same range as the one
        of the data.
        Input:
            clf: binary classifier with a method decision_function
            data: input data (X)
            num: discretization parameter
    """
    xmin, ymin = data.min(axis=0)
    xmax, ymax = data.max(axis=0)
    x, y = np.meshgrid(np.linspace(xmin, xmax, num), np.linspace(ymin, ymax))
    z = clf.decision_function(np.c_[x.ravel(), y.ravel()]).reshape(x.shape)
    cs = plt.contour(x, y, z, [0], colors='g')
    if label is not None:
        cs.levels = [label]
        plt.gca().clabel(cs)
    return cs
