import numpy as np
from scipy.linalg import eig
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting
# from mpl_toolkits import mplot3d

if __name__ == "__main__":

    X = np.loadtxt('data-dimred-X.csv', dtype=np.float, delimiter=',')
    y = np.loadtxt('data-dimred-y.csv', dtype=np.float)

    # X = X.T
    X = X-np.expand_dims(X.mean(axis=1),1)

    C = np.dot(X,X.T)/y.shape[0]
    l, U = eig(C)
    sorted_idxs = np.argsort(np.real(l))[::-1]
    l, U = l[sorted_idxs], U[:, sorted_idxs]


    # 2D PCA Projections
    dim = 2
    X_trans = np.dot(U.T[0:dim], X)

    plt.scatter(X_trans[0, y == 1], X_trans[1, y == 1], c='red', label='Class 1', alpha=0.75)
    plt.scatter(X_trans[0, y == 2], X_trans[1, y == 2], c='blue', label='Class 2', alpha=0.75)
    plt.scatter(X_trans[0, y == 3], X_trans[1, y == 3], c='green', label='Class 3', alpha=0.75)
    plt.title("PCA for dimension = 2 for 'data-dimred-X.csv'")
    plt.savefig("PCA_dim2.pdf", facecolor='w', edgecolor='w', papertype=None, format='pdf',
                transparent=False, bbox_inches='tight', pad_inches=0.1)
    plt.legend()
    plt.show()

    # 3D PCA Projections
    dim = 3
    X_trans = np.real(np.dot(U.T[0:dim], X))
    # print(X_trans)
    fig = plt.figure()
    ax = plt.axes(projection = '3d')
    ax.scatter(X_trans[0, y == 1], X_trans[1, y == 1], X_trans[2, y == 1], c='red', label='Class 1', alpha=0.75)
    ax.scatter(X_trans[0, y == 2], X_trans[1, y == 2], X_trans[2, y == 2], c='blue', label='Class 2', alpha=0.75)
    ax.scatter(X_trans[0, y == 3], X_trans[1, y == 3], X_trans[2, y == 3], c='green', label='Class 3', alpha=0.75)
    ax.set_title("PCA for dimension = 3 for 'data-dimred-X.csv'")
    plt.savefig("PCA_dim3.pdf", facecolor='w', edgecolor='w', papertype=None, format='pdf',
                transparent=False, bbox_inches='tight', pad_inches=0.1)
    ax.legend()
    plt.show()