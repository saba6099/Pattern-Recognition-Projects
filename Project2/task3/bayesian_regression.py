import numpy as np
from numpy.linalg import pinv, inv, norm
import matplotlib.pyplot as plt


class BayesianPoly():
    """
    (Well vectorized) implementation of Bayesian polynomial regression. Has a MLE module built-in for comparison.
    Predictive distribution module currently untested (not needed in prediction).
    """

    def __init__(self, reg_order, sigma_0_sq):
        """
        Initializes regression parameters.

        :param reg_order: The order of the regression.
        :param sigma_0_sq: The std of the prior on the labels
        """
        self.reg_order = reg_order
        self.sigma_0_sq = sigma_0_sq
        self.sigma = None
        self.chi = None
        self.mu = None
        self.X_mu = None
        self.X_sigma = None
        self.y_mu = None
        self.y_sigma = None
        self.W_map = None
        self.W_mle = None

    def fit(self, X, y):
        """
        Trains a MAP (as well as MLE for comparison) on Bayesian Regression. Internally standardizes data before
        MLE whereas unstandardized works fine for MAP.

        :param X: ndarray of training data
        :param y: ndarray of training labels
        :return:
        """
        # print(X, y)
        # Does some numpy array balancing among other things
        X = self.preprocess_X(X)    # Creates feature vector for polynomial x -> [1, x, x^2,... x^self.reg_order]
        y = self.preprocess_y(y)    # Numpy based dimension expansions

        if self.y_sigma is None:
            self.y_sigma = np.std(y)

        # Fitting the weights of the model using MAP
        self.W_map = np.dot(np.dot(inv(np.dot(X, X.T) + (np.identity(X.shape[0])*(self.y_sigma/self.sigma_0_sq)**2)),X),y)

        # Z-transform of data for MLE fit
        self.X_mu = np.mean(X, axis=1)
        self.X_sigma = np.std(X, axis=1)

        ztrans_X = ((X.T-self.X_mu)/(self.X_sigma+1e-8)).T
        ztrans_X[0,:] = 1.0 # Reinitializing w_0

        # Fitting the weights of the model using MLE
        self.W_mle = np.dot(np.dot(inv(np.dot(ztrans_X, ztrans_X.T)), ztrans_X), y)

        # Used only for predictive distribution estimation (NOT IMPLEMENTED PROPERLY OR TESTED CURRENTLY)
        self.chi = (np.dot(X, X.T)/(self.y_sigma**2)) + (np.identity(X.shape[0])/self.sigma_0_sq)
        self.mu = (np.dot(np.dot(inv(self.chi),X),y)/(self.y_sigma**2))

    def evaluate(self, X, w_type='map'):
        """
        Performs polynomial regression on a test set. Performs standardization if 'mle' results are required for
        comparison.
        :param X: ndarray of training data
        :param w_type: whether to use 'mle' or 'map' weights.
        :return: ndarray of predictions
        """
        X = self.preprocess_X(X)

        if w_type == 'map':
            y = np.dot(self.W_map.T, X).reshape(X.shape[1])
        elif w_type == 'mle':
            ztrans_X = ((X.T - self.X_mu) / (self.X_sigma+1e-8)).T
            ztrans_X[0, :] = 1.0
            # ztrans_X = np.log1p(X)
            y = np.dot(self.W_mle.T, ztrans_X).reshape(ztrans_X.shape[1])
        return y

    def evaluate_dist(self, X):
        """
        Evaluates the predictive distribution. NOT IMPLEMENTED CURRENTLY
        :param X:
        :return:
        """
        X = self.preprocess_X(X)
        print(X.shape)

        y = np.linspace(start=50, stop=120)
        print(y)
        y_pred = []
        fig, ax = plt.subplots(figsize=(10, 10))

        #y = np.array([])
        for x in X.T:
            x= np.expand_dims(x,1)
            mean = np.dot(self.mu.T, x)
            print(mean.shape)

            print(x.T.shape, inv(self.chi).shape, x.shape)
            std = (self.y_sigma**2) + np.dot(x.T,np.dot(inv(self.chi),x))

            print(self.y_sigma**2, std.shape)

            p_y_XD = norm.pdf(y, mean, std) # specify a normal distribution for sample mean and std
            print(p_y_XD.shape)
            for i in range(p_y_XD.shape[1]):
                print(x.shape, p_y_XD[0,i])
                ax.scatter(x[1,0], p_y_XD[0,i]*120)
            y_pred.append(p_y_XD)
        # print(y_pred)
        plt.show()
        return p_y_XD

    def preprocess_X(self, X):
        """
        Expands dimensions for data X and Creates the feature vector for polynomial regression
        :param X: ndarray of input data points
        :return: ndarray of transformed data points for polynomial regression of order = self.regr_order
        """
        def f(x):
            return np.array([x[0] ** i for i in range(self.reg_order + 1)])

        X = np.expand_dims(X, axis=1)
        X = np.apply_along_axis(f, axis=1, arr=X)
        X = X.T
        return X

    def preprocess_y(self, y):
        """
        Does simply numpy based dimension expansion on labels
        :param y: ndarray of labels
        :return: ndarray of labels with one extra dimension
        """
        y = np.expand_dims(y, axis=1)
        return y


def mse(y_pred, y_true):
    """
    Calculates the mean squared error of the prediction
    :param y_pred: ndarray of predicted ys
    :param y_true: ndarray of true ys
    :return: the mse of the predictions
    """
    return np.mean((y_pred-y_true)**2)
    # return y_pred-y_true


if __name__ == "__main__":

    # read data as 2D array of data type 'object'
    data = np.loadtxt('whData.dat', dtype=np.object, comments='#', delimiter=None)
    # Removing rows with missing weights
    data = data[data[:,0]!='-1',:]

    # read height data into 1D array (i.e. into a matrix)
    X = data[:, 1].astype(np.float)

    # read weight data into 1D array (i.e. into a vector)
    y = data[:, 0].astype(np.float)

    plt.scatter(X, y, color='black', label='Data')

    # print(X.shape, y.shape)

    model = BayesianPoly(reg_order=5, sigma_0_sq=3)
    model.fit(X, y)

    X_plot = np.linspace(150, 200, num=200)

    Y_plot = model.evaluate(X_plot, w_type='map')
    plt.plot(X_plot, Y_plot, 'k', linewidth=2, color='red', label='MAP') # Plot the gaussian

    Y_plot = model.evaluate(X_plot, w_type='mle')
    plt.plot(X_plot, Y_plot, 'k', linewidth=2, color='green', label='MLE')  # Plot the gaussian

    title = "Bayesian Regression Fit to: Height vs Weight"
    plt.title(title)
    plt.ylim(40,100)
    plt.ylabel("Weight")
    plt.xlabel("Height")
    plt.legend()
    plt.savefig("bayesian_regr.pdf", facecolor='w', edgecolor='w',
                papertype=None, format='pdf', transparent=False,
                bbox_inches='tight', pad_inches=0.1)
    plt.show()
    print("MSE for MAP estimation = {}".format(mse(model.evaluate(X, w_type='map'),y)))
    print("MSE for MLE estimation = {}".format(mse(model.evaluate(X, w_type='mle'), y)))