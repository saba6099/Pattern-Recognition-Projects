import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly

data = np.loadtxt('whData.dat', dtype=np.object, comments='#', delimiter=None)
# Removing rows with missing weights
saved_row = data[data[:, 0] == '-1', :]

data = data[data[:, 0] != '-1', :]
saved_heights = saved_row[:, 1].astype(np.float)
# read height data into 1D array (i.e. into a matrix)
hgt = data[:, 1].astype(np.float)

# read weight data into 1D array (i.e. into a vector)
wgt = data[:, 0].astype(np.float)

xmin = hgt.min()-15
xmax = hgt.max()+15
ymin = wgt.min()-15
ymax = wgt.max()+15

def plot_data_and_fit(h, w, x, y, plt_name):
    plt.plot(h, w, 'ko', x, y, 'r-')
    plt.title(plt_name)
    plt.ylabel("Weight")
    plt.xlabel("Height")
    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)
    plt.savefig("{}.pdf".format(plt_name), facecolor='w', edgecolor='w',
                papertype=None, format='pdf', transparent=False,
                bbox_inches='tight', pad_inches=0.1)
    plt.show()

def trsf(x):
    return x / 100.

n = 10
x = np.linspace(xmin, xmax, 100)

# method 1:
# regression using ployfit
c = poly.polyfit(hgt, wgt, n)
y = poly.polyval(x, c)
plot_data_and_fit(hgt, wgt, x, y, "Result with method1")

# method 2:
# regression using the Vandermonde matrix and pinv
X = poly.polyvander(hgt, n)
c = np.dot(la.pinv(X), wgt)
y = np.dot(poly.polyvander(x,n), c)
plot_data_and_fit(hgt, wgt, x, y, "Result with method2")

# method 3:
# regression using the Vandermonde matrix and lstsq
X = poly.polyvander(hgt, n)
c = la.lstsq(X, wgt)[0]
y = np.dot(poly.polyvander(x,n), c)
plot_data_and_fit(hgt, wgt, x, y, "Result with method3")

# method 4:
# regression on transformed data using the Vandermonde
# matrix and either pinv or lstsq
X = poly.polyvander(trsf(hgt), n)
c = np.dot(la.pinv(X), wgt)
y = np.dot(poly.polyvander(trsf(x),n), c)
plot_data_and_fit(hgt, wgt, x, y, "Result with method4")