# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as la
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt

data = np.loadtxt("whData.dat", dtype=np.object, comments='#',  delimiter=None)
#removing outliers or data with weight -1 
data = data[ data[:,0] != '-1' , : ] 

hgt = data[:,1].astype( np.float)
wgt = data[:,0].astype( np.float)

xmin = hgt.min()-15
xmax = hgt.max()+15
ymin = wgt.min()-15
ymax = wgt.max()+15

def plot_data_and_fit(h, w, x, y, figname):
    plt.plot(h, w, 'ko', x, y, 'r-')
    plt.xlabel('Height')
    plt.ylabel('Weight')
    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)
    plt.savefig("{}.pdf".format(figname), facecolor='w', edgecolor='w',
                 orientation='potrait', format='pdf', transparent=False)
    plt.show()
def trsf(x):
    return x / 100.

n = 10
x = np.linspace(xmin, xmax, 100)

# method 1:
# regression using polyfit
c = poly.polyfit(hgt, wgt, n)
y = poly.polyval(x, c)
plot_data_and_fit(hgt, wgt, x, y, "polyfit")

# method 2:
# regression using the Vandermonde matrix and pinv
X = poly.polyvander(hgt, n)
#Pattern Recognition (1) B-IT, Summer Term 2019
c = np.dot(la.pinv(X), wgt)
y = np.dot(poly.polyvander(x,n), c)
plot_data_and_fit(hgt, wgt, x, y, "method2")

# method 3:
# regression using the Vandermonde matrix and lstsq
X = poly.polyvander(hgt, n)
c = la.lstsq(X, wgt)[0]
y = np.dot(poly.polyvander(x,n), c)
plot_data_and_fit(hgt, wgt, x, y, "method3")

# method 4:
# regression on transformed data using the Vandermonde
# matrix and either pinv or lstsq
X = poly.polyvander(trsf(hgt), n)
c = np.dot(la.pinv(X), wgt)
y = np.dot(poly.polyvander(trsf(x),n), c)
plot_data_and_fit(hgt, wgt, x, y, "method4")
