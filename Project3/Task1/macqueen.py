import numpy as np
import matplotlib.pyplot as plt
from numpy import vstack, array
from pylab import plot, show, savefig
import csv
import random
from scipy.cluster.vq import kmeans2, vq
# import numpy_indexed as npi
import time

if __name__ == "__main__":

    # read data as 2D array of data type 'np.float'
    result = np.array(list(csv.reader(open("data-clustering-2.csv", "r"), delimiter=","))).astype("float")

    ti = []
    colors = ['red', 'green', 'blue']#, 'cyan', 'orange']
    result = result.T
    classes = np.zeros(result.shape[0])

    mui = np.random.rand(len(colors), 2)
    ni = np.zeros((len(colors),))
    for j in range(result.shape[0]):
        start = time.time()

        loss = np.square(np.linalg.norm(mui - result[j, :], axis=1))
        iw = np.argmin(loss)

        ni[iw] += 1
        mui[iw] += (result[j, :] - mui[iw]) / ni[iw]

        classes[j] = iw
        norms = np.zeros((j+1 , len(colors)))
        for c in range(len(colors)):
            norms[:, c] = np.square(np.linalg.norm(result[0:j+1, :] - mui[c, :], axis=1))
        classes[0:j+1] = np.argmin(norms, axis=1)

        end = time.time()
        time_taken = end - start
        ti.append(time_taken)

    print("time_taken")
    print(sum(ti) / len(ti))

    plt.title("McQueen")
    plt.xlim(result[:,0].min() - 1, 1 + result[:,0].max())
    plt.ylim(result[:,1].min() - 1, 1 + result[:,1].max())
    plt.scatter(result[:,0], result[:,1], c=[colors[i] for i in classes.astype(int)], s=5.5,
               label='Points')
    plt.scatter(mui[:, 0], mui[:, 1], c=colors, s=40,
                   marker='*', label='Centroids')
    plt.savefig("mcqueen-2.pdf", facecolor='w', edgecolor='w',
                papertype=None, format='pdf', transparent=False,
                bbox_inches='tight', pad_inches=0.1)
    plt.savefig("mcqueen-2.png", facecolor='w', edgecolor='w',
                papertype=None, format='png', transparent=False,
                bbox_inches='tight', pad_inches=0.1)
    plt.legend()
    plt.show()
