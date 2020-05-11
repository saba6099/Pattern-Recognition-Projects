import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting
# from mpl_toolkits import mplot3d

if __name__ == "__main__":

    X = np.loadtxt('data-dimred-X.csv', dtype=np.float, delimiter=',')
    y = np.loadtxt('data-dimred-y.csv', dtype=np.float)

    S_w = np.zeros((X.shape[0], X.shape[0]), dtype=np.float)
    S_b = np.zeros((X.shape[0], X.shape[0]), dtype=np.float)
    mu = np.expand_dims(np.mean(X,axis=1), 1)

    for cl in np.unique(y):
        X_cl = X[:,y==cl]
        mu_cl  = np.expand_dims(np.mean(X_cl, axis=1),1)
        covar_cl_w = np.dot((X_cl - mu_cl), (X_cl - mu_cl).T)/X_cl.shape[1]
        covar_cl_b = np.dot((X_cl - mu), (X_cl - mu).T)/X_cl.shape[1]
        S_w = S_w + covar_cl_w
        S_b = S_b + covar_cl_b

    l, U = eig(np.dot(np.linalg.pinv(S_w),S_b))
    sorted_idxs = np.argsort(np.real(l))[::-1]
    # print(sorted_idxs)
    l, U = l[sorted_idxs], U[:, sorted_idxs]

    # 2D Multiclass LDA Projections
    dim = 2
    X_trans = np.real(np.dot(U.T[0:dim], X))

    plt.scatter(X_trans[0, y == 1], X_trans[1, y == 1], c='red', label='Class 1', alpha=0.75)
    plt.scatter(X_trans[0, y == 2], X_trans[1, y == 2], c='blue', label='Class 2', alpha=0.75)
    plt.scatter(X_trans[0, y == 3], X_trans[1, y == 3], c='green', label='Class 3', alpha=0.75)
    plt.title("LDA for dimension = 2 for 'data-dimred-X.csv'")
    plt.savefig("LDA_dim2.pdf", facecolor='w', edgecolor='w', papertype=None, format='pdf',
                transparent=False, bbox_inches='tight', pad_inches=0.1)
    plt.legend()
    plt.show()

    # 3D Multiclass LDA Projections
    dim = 3
    X_trans = np.real(np.dot(U.T[0:dim], X))

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(X_trans[0, y == 1], X_trans[1, y == 1], X_trans[2, y == 1], c='red', label='Class 1', alpha=0.75)
    ax.scatter(X_trans[0, y == 2], X_trans[1, y == 2], X_trans[2, y == 2], c='blue', label='Class 2', alpha=0.75)
    ax.scatter(X_trans[0, y == 3], X_trans[1, y == 3], X_trans[2, y == 3], c='green', label='Class 3', alpha=0.75)
    ax.set_title("LDA for dimension = 3 for 'data-dimred-X.csv'")
    plt.savefig("LDA_dim3.pdf", facecolor='w', edgecolor='w', papertype=None, format='pdf',
                transparent=False, bbox_inches='tight', pad_inches=0.1)
    ax.legend()
    plt.show()