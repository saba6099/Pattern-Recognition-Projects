import numpy.linalg as la
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt('whData.dat', dtype=np.object, comments='#', delimiter=None)

# Removing rows with missing weights
data = data[data[:,0]!='-1',:]

#Reading weight and height data
hgt = data[:, 1].astype(np.float)
wgt = data[:, 0].astype(np.float)

xmin = hgt.min()-15
xmax = hgt.max()+15
ymin = wgt.min()-15
ymax = wgt.max()+15

def plot_data_and_fit(h, w, x, y, plt_name):
	plt.plot(h, w, 'ko', x, y, 'r-')
	plt.xlim(xmin,xmax)
	plt.ylim(ymin,ymax)
	plt.ylabel("Weight")
	plt.xlabel("Height")
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
plot_data_and_fit(hgt, wgt, x, y, "Method1-result")

# method 2:
# regression using the Vandermonde matrix and pinv
X = poly.polyvander(hgt, n)
c = np.dot(la.pinv(X), wgt)
y = np.dot(poly.polyvander(x,n), c)
plot_data_and_fit(hgt, wgt, x, y, "Method2-result")

# method 3:
# regression using the Vandermonde matrix and lstsq
X = poly.polyvander(hgt, n)
c = la.lstsq(X, wgt)[0]
y = np.dot(poly.polyvander(x,n), c)
plot_data_and_fit(hgt, wgt, x, y, "Method3-result")

# method 4:
# regression on transformed data using the Vandermonde
# matrix and either pinv or lstsq
X = poly.polyvander(trsf(hgt), n)
c = np.dot(la.pinv(X), wgt)
y = np.dot(poly.polyvander(trsf(x),n), c)
plot_data_and_fit(hgt, wgt, x, y, "Method4-result")
