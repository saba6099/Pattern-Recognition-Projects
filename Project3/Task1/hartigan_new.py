import numpy as np
import matplotlib.pyplot as plt
from numpy import vstack, array
from pylab import plot, show, savefig
import csv
import random
from scipy.cluster.vq import kmeans2, vq
import numpy_indexed as npi
import time

if __name__ == "__main__":

    # read data as 2D array of data type 'np.float'
    result = np.array(list(csv.reader(open("data-clustering-2.csv", "rb"), delimiter=","))).astype("float")
    ti = []
    colors = ['red', 'green', 'blue', 'cyan', 'orange']
    X = result[0, :]
    Y = result[1, :]
    result = result.T
    k = 0
    while (k < 10):
        start = time.time()
        Ct = np.hstack(
            (result, np.reshape(np.random.choice(range(0, 3), result.shape[0], replace=True), (result.shape[0], 1))))
        meu = (npi.group_by(Ct[:, 2]).mean(Ct))[1][:, 0:2]
        Converged = False
        while Converged is False:
            Converged = True
            for j in range(0, Ct.shape[0]):
                Cj = Ct[j, 2]
                dmin = []
                for i in range(0, 3):
                    Ct[j, 2] = i
                    G = (npi.group_by(Ct[:, 2])).split(Ct)
                    dist = 0
                    # print(G)
                    for p in range(0, 3):
                        t = (G[p][:, 0:2])
                        mi = np.reshape(np.mean(t, axis=0), (1, 2))
                        t = np.sum((t - mi) ** 2, axis=1)
                        dist = dist + np.sum(t, axis=0)
                    dmin.append(dist)
                Cw = np.argmin(dmin)
                if Cw != Cj:
                    Converged = False
                    Ct[j, 2] = Cw
                    meu = (npi.group_by(Ct[:, 2]).mean(Ct))[1][:, 0:2]
                else:
                    Ct[j, 2] = Cj
        end = time.time()
        time_taken = end - start
        ti.append(time_taken)
        k = k + 1
    print("time_taken")
    print(sum(ti) / len(ti))

    meu = np.hstack((meu, np.reshape(np.array(list(range(3))), (3, 1))))
    cp = Ct
    plt.title("Hartigan")
    plt.xlim(xmin=np.min(Ct[:, 0]) - 1, xmax=np.max(1 + Ct[:, 0]))
    plt.ylim(ymin=np.min(Ct[:, 1]) - 1, ymax=np.max(1 + Ct[:, 1]))
    plt.scatter(Ct[:, 0], Ct[:, 1], c=[colors[i] for i in (Ct[:, 2]).astype(int)], s=5.5,
               label='Points')
    plt.scatter(meu[:, 0], meu[:, 1], c=[colors[i] for i in (meu[:, 2]).astype(int)], s=40,
                   marker='*', label='Centroids')
    plt.savefig("Hartigan Clustering-2.pdf", facecolor='w', edgecolor='w',
                papertype=None, format='pdf', transparent=False,
                bbox_inches='tight', pad_inches=0.1)
    plt.legend()
    plt.show()
