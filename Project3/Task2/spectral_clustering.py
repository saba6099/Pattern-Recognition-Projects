import numpy as np
from numpy.linalg import norm, eig
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # read data as 2D array of data type 'np.float'
    X = np.loadtxt('data-clustering-2.csv', dtype=np.float, delimiter=',')
    X = X.T

    beta = 1.0
    sim = np.empty((X.shape[0], X.shape[0]), dtype=np.float)
    diag = np.zeros((X.shape[0], X.shape[0]), dtype=np.float)

    for i in range(X.shape[0]):
        sim[i,:] = np.exp(-1.0*beta*np.power(norm(X[i]-X, ord=2, axis=1),2))
        diag[i,i] = np.sum(sim[i,:])

    # Calculating Laplacian
    L = diag - sim

    # Sorting Eigen Values and Vectors
    w, v = eig(L)
    sorted_idxs = np.argsort(np.real(w))
    w, v = w[sorted_idxs], v[:, sorted_idxs]

    # Fiedler Vector and clustering into {0,1} (represented in plot as {1,2})
    fv = v.T[1]
    y = (fv>0)*1
    print(y)

    color_list = ['blue', 'red']
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', label='Class 1', alpha=0.75)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', label='Class 2', alpha=0.75)
    plt.title("Spectral Clustering for Beta = {}".format(beta))
    plt.savefig("spectral_clustering_Beta{}.pdf".format(beta), facecolor='w', edgecolor='w', papertype=None, format='pdf',
                transparent=False, bbox_inches='tight', pad_inches=0.1)
    plt.legend()
    plt.show()

    # from sklearn.cluster import KMeans, SpectralClustering
    # kmeans = SpectralClustering(n_clusters=2).fit(X)
    #
    # color_list = ['blue', 'red']
    # plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap=ListedColormap(color_list), alpha=0.75)
    # plt.show()