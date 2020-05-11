import numpy as np
import matplotlib.pyplot as plt
from numpy import vstack,array
from pylab import plot,show,savefig
import csv
from scipy.cluster.vq import kmeans,vq
import time

if __name__ == "__main__":

    # read data as 2D array of data type 'np.float'
    result = np.array(list(csv.reader(open("data-clustering-1.csv", "rb"), delimiter=","))).astype("float")
    t=[]
    X = result[0,:]
    Y = result[1,:]
    plt.scatter(X,Y)
    plt.savefig("Scatter plot of data.pdf", facecolor='w', edgecolor='w',
                papertype=None, format='pdf', transparent=False,
                bbox_inches='tight', pad_inches=0.1)
    result=result.T
    plt.show()
    i=0
    while(i<10):
        start = time.time()
        centroids, labels = kmeans(result, 3)  #assign each sample to cluster random
        idx, _ = vq(result, centroids)
        end = time.time()
        time_taken = end - start
        t.append(time_taken)
        i = i + 1
    print("time_taken")
    print(sum(t) / len(t))

    # some plotting using numpy's logical indexing
    plot(result[idx == 0, 0], result[idx == 0, 1], 'ob',
         result[idx == 1, 0], result[idx == 1, 1], 'or',
         result[idx == 2, 0], result[idx == 2, 1], 'og')  # third cluster points
    plot(centroids[:, 0], centroids[:, 1], 'sm', markersize=8)

    savefig("Lloyd Clustering.pdf")
    show()