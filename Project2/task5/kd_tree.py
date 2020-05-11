import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

from sklearn.neighbors import KDTree as KDTREE
from sklearn.model_selection import ParameterGrid


import time

class KDTree:
    """
    KDTRee object: designed for NN search but can run generically
    """
    def __init__(self, k, dim_mode='alternate', split_mode='mid'):
        """
        Initializes various parameters of the KDTree object

        :param k: The number of dimensions of the data
        :param dim_mode: The method of selecting dimensions for splitting. Must be in ['alternate', 'var']
        :param split_mode: The method of selecting split points. Must be in ['mean', 'median']
        """
        self.k = k
        self.X = None
        self.Y = None
        self.root = None
        self.dim_mode = dim_mode
        self.split_mode = split_mode
        self.max_depth = 0                      # Stores the maximum depth of the Tree

    def fit(self, x, y=None, depth=None):
        """
        :param x: ndarray of training data
        :param y: ndarray of training labels. Can be provided to draw a grouped plot.
        :return: the root node of KDTree as a KDNode object
        """
        self.X = x
        self.Y = y
        self.depth = depth
        idxs = np.array([i for i in range(self.X.shape[0])])

        self.root = self._fit(idxs=idxs, depth=0, last_dim=None)

    def _fit(self, idxs, depth, last_dim = None):
        """
        Recursive internal method for fitting. Creates a leaf node if the has one point index in idxs

        :param idxs: indices of main training data used in current node
        :param depth: depth of the tree currently
        :param last_dim: last dimension used for splitting if self.dim_mode is 'alternate'
        :return: leaf or internal node of kDtree as KDNode object
        """
        if depth>self.max_depth:
            self.max_depth = depth

        if self.depth is not None:
            if depth==self.depth:
                return KDNode(split_dim=None, val=None, idxs=idxs)
        else:
            # print(idxs.shape[0])
            if idxs.shape[0]==1:
                return KDNode(split_dim=None, val=None, idxs=idxs)

        dim = self.select_dim(idxs=idxs if self.dim_mode == 'var' else None, last_dim=last_dim)
        split_point = self.select_split(split_dim=dim, idxs=idxs)

        # Only if there is a split to be made
        node = KDNode(split_dim=dim, val=split_point)

        left_idxs = idxs[self.X[idxs,dim]<=split_point]
        node.left_node = self._fit(idxs=left_idxs, depth=depth+1, last_dim=dim) if left_idxs.shape[0]>0 else None

        right_idxs = idxs[self.X[idxs, dim] > split_point]
        node.right_node = self._fit(idxs=right_idxs, depth=depth+1, last_dim=dim) if right_idxs.shape[0] > 0 else None

        return node

    def select_dim(self, idxs=None, last_dim=None):
        """
        Selects the split dimension based on self.dim_mode

        :param idxs: indices of main training data used in current node
        :param last_dim: the last dimension used if self.dim_mode is 'alternate'
        :return:
        """
        if self.dim_mode == 'alternate':
            if last_dim == None:
                return 0
            return (last_dim+1)%self.k

        elif self.dim_mode == 'var': # For dim of highest variance
            return np.argmax(np.var(self.X[idxs], axis=0))
        else:
            raise ValueError # Choose 'alternate' or 'var' for dimension choice

    def select_split(self, split_dim, idxs):
        """
        Returns the next split point to use based on chosen split mode.

        :param split_dim: The dimension to be used for splitting
        :param idxs: indices of main training data used in current node
        :return: the split point on dimension = 'split_dim' of data
        """
        if self.split_mode == 'mean':
            return np.mean(self.X[idxs, split_dim], axis=0)
        elif self.split_mode == 'median':
            return np.median(self.X[idxs, split_dim], axis=0)
        else:
            raise ValueError # Choose split point as 'mid' or 'median' point of data

    def evaluate(self, X):
        """
        Evaluates and returns the nearest neighbour distances and nearest neighbour indices of X based on self.X
        :param X: ndarray of test data
        :return: (ndarray: best distances, ndarray: best indices)
        """
        def f(x):
            return self._evaluate(x, self.root, best=None, best_idx=None)
        dists_idxs = np.apply_along_axis(f, axis=1, arr=X)
        #idxs = dists_idxs[:,1]
        return dists_idxs[:,0], dists_idxs[:,1]

    def _evaluate(self, x, node, best, best_idx):
        """
        Internal recursive evaluator. Don't call this. call evaluate(..)
        :param x: ndarray of single data point
        :param node: current node
        :param best: best distance till now
        :param best_idx: index of best point till now from training set
        :return: best, best_idx
        """
        if node.idxs is not None: # Leaf Node Condition
            # print(self.X[node.idxs], x)
            new_best = np.linalg.norm(x-self.X[node.idxs], ord=2)
            if best is None or best < new_best:
                best = new_best
                best_idx = node.idxs
        else:                     # Intermediate Node Condition
            if x[node.split_dim]<= node.val: # Fancy way of taking care of initial traversal
                search_first = 'left'
            else:
                search_first = 'right'

            if search_first == 'left':
                if best == None or (x[node.split_dim]-best<=node.val):
                    best, best_idx = self._evaluate(x, node.left_node, best, best_idx)
                elif best == None or (x[node.split_dim]-best>node.val):
                    best, best_idx = self._evaluate(x, node.right_node, best, best_idx)
            else:
                if best == None or (x[node.split_dim]-best>node.val):
                    best, best_idx = self._evaluate(x, node.right_node, best, best_idx)
                elif best == None or (x[node.split_dim]-best<=node.val):
                    best, best_idx = self._evaluate(x, node.left_node, best, best_idx)

        return best, best_idx

    def plot(self, savefig=False):
        """
        Plots the kDTree if k=2
        :param savefig: saves the p[ot on 'True'
        :return:
        """
        assert self.k == 2
        color_list = ['blue', 'red']

        fig = plt.figure()
        ax = fig.add_subplot(111)

        xmin, xmax, ymin, ymax = min(self.X[:, 0]), max(self.X[:, 0]), min(self.X[:, 1]), max(self.X[:, 1])
        if self.Y is not None:
            plt.scatter(self.X[:, 0], self.X[:, 1], c=self.Y, cmap=ListedColormap(color_list), alpha=0.75)
        else:
            plt.scatter(self.X[:, 0], self.X[:, 1], color='blue', alpha=0.75)
        plt.xlim((xmin, xmax))
        plt.ylim((ymin, ymax))
        plt.ylabel("Weight")
        plt.xlabel("Height")
        self._plot(self.root, ax, xmin, xmax, ymin, ymax)
        plt.title("kD-Tree for Split Point Selection: {} \n Split Dim. selection: {}".format(self.split_mode, self.dim_mode))
        if savefig:
            plt.savefig("kdtree_splitmode_{}_dimmode_{}.pdf".format(self.split_mode, self.dim_mode), facecolor='w', edgecolor='w',
                        papertype=None, format='pdf', transparent=False,
                        bbox_inches='tight', pad_inches=0.1)
        plt.show()

    def _plot(self, node, ax, xmin, xmax, ymin, ymax):
        """
        Internal recursive plotting function. Do not use this. Use plot(..)
        """
        if node.split_dim is not None:
            dim = node.split_dim
            val = node.val
            if dim == 0:
                # y_plt = np.linspace(ymin, ymax, int((ymin-ymax)*interval_frac))
                # x_plt = np.ones_like(y_plt)*val
                left_xmin, left_xmax, right_xmin, right_xmax = xmin, val, val, xmax
                left_ymin, left_ymax, right_ymin, right_ymax = ymin, ymax, ymin, ymax
                line = Line2D([val, val], [ymin, ymax], c='black', linewidth=1.5)

            else:
                # x_plt = np.linspace(xmin, xmax, int((xmax - xmin) * interval_frac))
                # y_plt = np.ones_like(x_plt) * val
                left_xmin, left_xmax, right_xmin, right_xmax = xmin, xmax, xmin, xmax
                left_ymin, left_ymax, right_ymin, right_ymax = ymin, val, val, ymax
                line = Line2D([xmin, xmax], [val, val], c='black', linewidth=1.5)
            ax.add_line(line)

            #plt.scatter(x_plt, y_plt, color='black', linestyle='-', linewidth=0.1)
            self._plot(node.left_node, ax, left_xmin, left_xmax, left_ymin, left_ymax)
            self._plot(node.right_node, ax, right_xmin, right_xmax, right_ymin, right_ymax)

class KDNode:
    """
    Node object for kDTree. Holds important info for kDTree creation
    """
    def __init__(self, split_dim, val, idxs=None):

        self.split_dim = split_dim
        self.val = val
        self.idxs = idxs # Only populated for leaf nodes
        self.left_node = None
        self.right_node = None


def accuracy(preds, true):
    return np.sum((preds==true)*1)/preds.shape[0]


if __name__ == "__main__":

    # Training and Testing Created KDTree object
    print("Training created KDTree object")
    # INITIALIZING DIFFERENT COMBINATIONS OF SPLIT MODES AND DIMENSION SELECTION MODES USING
    # sklearn.model_selection.ParameterGrid
    split_mode = ["mean", "median"]
    dim_mode = ["alternate", "var"]

    params = ParameterGrid({'split_mode': split_mode, 'dim_mode': dim_mode})

    # Training and Testing for every parameter combination in 'params'.
    for p in params:
        sm = p["split_mode"]
        dm = p["dim_mode"]

        print("Split Mode: {}, Dimension Mode: {}".format(sm, dm))

        # Loading training Data
        with open("data2-train.dat", "r") as f:
            d = np.loadtxt(f)

        X = d[:, 0:2]
        train_Y = d[:, 2]
        train_Y[train_Y == -1] = 0
        print("Number of Samples in training set: {}".format(X.shape[0]))

        # Training and plotting using KDTree object
        t1 = time.time()
        model = KDTree(k=2, split_mode=sm, dim_mode=dm) # Creating KDTree object
        model.fit(x=X, y=None, depth=None)                          # Training
        t2 = time.time()
        print("Time taken for k-d tree creation: {} secs".format(t2-t1))
        print('Depth of Tree created: {}'.format(model.max_depth))

        model.plot(savefig=True)                                 # generating plot

        # Loading Test Set
        with open("data2-test.dat", "r") as f:
            d = np.loadtxt(f)

        X = d[:,0:2]
        Y = d[:,2]
        Y[Y==-1] = 0
        print("\nNumber of Samples in test set: {}".format(X.shape[0]))

        t1 = time.time()
        dists, idxs = model.evaluate(X)                              # Evaluating trained KDTree
        Y_pred = train_Y[idxs.astype(np.int)]
        t2 = time.time()
        print("Time taken for k-d tree querying for 1-NN on data2-test.dat: {} secs".format(t2 - t1))
        print("Accuracy on Test Set for KDTree = {}\n\n".format(accuracy(Y_pred, Y)))
    # Manual KDTree coding done


    # TESTING SKLEARN KDTREE FOR BENCHMARKING
    print("\n\nTesting with SKlearn KDTree.")
    with open("data2-train.dat", "r") as f:
        d = np.loadtxt(f)

    X = d[:,0:2]
    train_Y = d[:,2]
    train_Y[train_Y==-1] = 0
    print("Number of Samples in training set: {}".format(X.shape[0]))

    t1 = time.time()
    sklearn_tree = KDTREE(X, leaf_size=1)
    t2 = time.time()
    print("Time taken for Sklearn k-d tree creation: {} secs".format(t2-t1))

    with open("data2-test.dat", "r") as f:
        d = np.loadtxt(f)

    X = d[:, 0:2]
    Y = d[:, 2]
    Y[Y == -1] = 0
    print("\nNumber of Samples in test set: {}".format(X.shape[0]))

    t1 = time.time()
    dist, idx = sklearn_tree.query(X)
    t2 = time.time()
    sk_preds = train_Y[idx].reshape(idx.shape[0])

    print("Time taken for Sklearn k-d tree querying for 1-NN on data2-test.dat: {} secs".format(t2 - t1))

    print("Accuracy on Test Set for Sklearn KDTree = {}".format(accuracy(sk_preds, Y)))
    # SKLEARN TESTS DONE