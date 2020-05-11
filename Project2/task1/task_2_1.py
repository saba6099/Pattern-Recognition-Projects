import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt

mu = None
std = None
def standardize( data):
    """
    Peforms standardization (Z-transform) of data and stores mu and std for use during evaluation
    in global variables
    """
    global  mu, std
    mu = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    data = (data - mu) / (std+1e-8)
    data[:,0] = 1.0
    return data

def fit(x,y, order=1):
    """
    Fits the data after preprocessing and standardization to a polynomial of degree 'order'.
    Program is solved using np.linalg.inv
    """
    d = {}
    d['x' + str(0)] = np.ones([1, len(x)],dtype=np.float)[0]
    for i in np.arange(1, order + 1):
        d['x' + str(i)] = x ** i
    X = np.column_stack(d.values())
    X = standardize(X)
    inter1 = np.matmul(np.transpose(X), X)
    #find inverse
    theta = np.matmul(np.matmul(linalg.inv(inter1), np.transpose(X)), y)
    return theta


def evaluate(saved_heights,theta):
    """
    Performs polynomial regression based on the saved weights after standardizing the data.
    """
    py=[]
    for i in saved_heights:
        r= theta[0]
        for j in range(1, len(theta)):
            r = r + (theta[j] * (pow(i,j)-mu[j])/(std[j]+1e-8))
        py.append(r)
    return py


data = np.loadtxt('whData.dat', dtype=np.object, comments='#', delimiter=None)
# Removing rows with missing weights
saved_row = data[data[:, 0] == '-1', :]

data = data[data[:, 0] != '-1', :]
saved_heights = saved_row[:, 1].astype(np.float)
# read height data into 1D array (i.e. into a matrix)
X = data[:, 1].astype(np.float)

# read weight data into 1D array (i.e. into a vector)
y = data[:, 0].astype(np.float)

d = [1,5,10]
for i in d:

    theta = fit(X,y,i)
    h = np.linspace(155,200, 100)
    #plot_predictedPolyLine(h,y,theta)

    px = evaluate(h,theta)
    y_missing = evaluate(saved_heights,theta)
    print("\nMissing Values:\n(Heights, Weights for d=",i,")\n----------")
    for i in zip(saved_heights, y_missing):
        print(i)
    plt.scatter(saved_heights, y_missing, color='black', label='missing')
    plt.scatter(X,y , color='blue', label='data')
    plt.plot(h, px, color='red', label='MLE fit')
    plt.title('Height vs Weight using Least Squares Method') 
    plt.ylim((30,100))
    plt.xlabel('Height')
    plt.ylabel('Weight')
    plt.legend()
    plt.savefig("mle_poly_regr.pdf", facecolor='w', edgecolor='w',
                    papertype=None, format='pdf', transparent=False,
                    bbox_inches='tight', pad_inches=0.1)
    plt.show()