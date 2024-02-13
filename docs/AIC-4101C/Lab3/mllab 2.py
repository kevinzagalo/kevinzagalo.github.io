"""
Packages:
    numpy as np
    matplotlib.pyplot as plt

Functions:
    plotXY
    plot_frontiere
    map_regions
"""

print(__doc__)

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def plotXY(X, Y, legend=True, ax=None):
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
        XX = np.concatenate((X, Y), axis=0)
        YY = np.concatenate((np.ones(X.shape[0]), -np.ones(Y.shape[0])))
    else:
        XX = X
        YY = Y
    for icl, cl in enumerate(np.unique(YY)):
    	if ax is not None:
    		ax.scatter(XX[YY==cl, 0], XX[YY==cl, 1], label='Class {0:d}'.format(icl+1))
    	else:
        	plt.scatter(XX[YY==cl, 0], XX[YY==cl, 1], label='Class {0:d}'.format(icl+1))
    if ax is not None:
    	ax.axis('equal')
    	ax.legend()
    	return ax
    else:
    	plt.axis('equal')
    	plt.legend()



def plot_frontiere(clf, data=None, num=500, ax=None):
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
    if ax is not None:
    	ax.contour(x, y, z, [0], colors='g')
    	return ax
    else:
       	plt.contour(x, y, z, [0], colors='g')


def map_regions(clf, data=None, num=500, ax=None):
    """
        Map the regions f(x)=1…K of the classifier clf within the same range as the one
        of the data.
        Input:
            clf: classifier with a method predict
            data: input data (X)
            num: discretization parameter
    """
    xmin, ymin = data.min(axis=0)
    xmax, ymax = data.max(axis=0)
    x, y = np.meshgrid(np.linspace(xmin, xmax, num), np.linspace(ymin, ymax))
    z = clf.predict(np.c_[x.ravel(), y.ravel()]).reshape(x.shape)
    zmin, zmax = z.min(), z.max()
    if ax is not None:
        ax.imshow(z, origin='lower', interpolation="nearest",
            extent=[xmin, xmax, ymin, ymax], cmap=cm.coolwarm,
            alpha=0.3)
        return ax
    else:
        plt.imshow(z, origin='lower', interpolation="nearest",
            extent=[xmin, xmax, ymin, ymax], cmap=cm.coolwarm,
            alpha=0.3)

