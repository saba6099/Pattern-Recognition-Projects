import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm

def bivariate_norm_distr(X,Y):
    """
    Evaluates on a bivariate gaussian distribution
    :param X: ndarray (pass heights)
    :param Y: ndarray (pass weights)
    :return: ndarray of probabilities for (x,y) in zip(X,Y)
    """
    std_x = std_h
    std_y = std_w
    c = np.sqrt(1 - rho ** 2) * 2 * np.pi * std_x * std_y
    c = 1 / c
    ret = []
    for x, y in zip(X,Y):

        q = ((y/std_y)**2)/2 + (((x - rho*(std_x/std_y)*y)**2)/(2*(1-rho**2)*std_x**2))
        ret.append(c*np.exp(-q))
    return np.array(ret)


def biv_norm_condn_exp(X):
    """
    Calculates E[Y|x0] for a bivariate gaussian distribution
    :param X: Values to be evaluated
    :return: ndarray of condiitional expectations
    """
    std_x = std_h
    std_y = std_w
    mu_x = mu_h
    mu_y = mu_w
    # print(X)
    def f(x):
        return mu_y + rho*(std_y/std_h)*(x-mu_x)
    E_y_x = np.apply_along_axis(f, 0, X)
    return E_y_x


if __name__ == "__main__":
    # read data as 2D array of data type 'object'
    data = np.loadtxt('whData.dat', dtype=np.object, comments='#', delimiter=None)
    # Removing rows with missing weights

    data_clean = data[data[:, 0] != '-1', :]

    # read weight data into 1D array (i.e. into a vector)
    W = data_clean[:, 0].astype(np.float)

    # read height data into 1D array (i.e. into a matrix)
    H = data_clean[:, 1].astype(np.float)

    rho = np.corrcoef(H, W)[0,1]

    std_h = np.std(H)
    std_w = np.std(W)

    mu_h = np.mean(H)
    mu_w = np.mean(W)

    print("Parameters of Bivariate Gaussian Distribution")
    print("Correlation={}, std_X = {}, std_W = {}, mu_X = {}, mu_Y={}".format(rho, std_h, std_w, mu_h, mu_w))

    fig, ax = plt.subplots()

    # Creating contour representing distribution
    # Note: Joint distribution formulas require zero mean
    N = 1000
    x_h = np.linspace(140, 200, N)
    y_w = np.linspace(40, 100, N)

    H, W = np.meshgrid(x_h, y_w)

    W = (W - mu_w)# / std_w
    H = (H - mu_h)# / std_h

    Z = bivariate_norm_distr(H, W)
    # Z = (Z-Z.mean())/(Z.std()+1e-18)
    # print(np.max(Z))

    W = (W + mu_w)
    H = H + mu_h
    cs = ax.contourf(H, W, Z, cmap='viridis', alpha=0.6)
    cbar = fig.colorbar(cs)
    plt.ylabel("Weight")
    plt.xlabel("Height")

    # Plotting the data and missing values later to have proper visualization
    # read weight data into 1D array (i.e. into a vector)
    W = data_clean[:, 0].astype(np.float)

    # read height data into 1D array (i.e. into a matrix)
    H = data_clean[:, 1].astype(np.float)

    ax.scatter(H,W, color='black', label='data')

    # Formula for conditional expectation seems to not require zero-mean values
    missing_H = data[data[:, 0] == '-1', 1].astype(np.float)
    missing_W = biv_norm_condn_exp(missing_H)

    ax.scatter(missing_H,missing_W, color='red', label='missing')

    print("Missing Data Prediction:")
    print("Weights and Heights:")
    for i in zip(missing_W, missing_H):
        print(i)

    plt.legend()
    plt.title("Bivariate Gaussian Distribution with missing value prediction")
    plt.savefig("biv_gaussian.pdf", facecolor='w', edgecolor='w',
                papertype=None, format='pdf', transparent=False,
                bbox_inches='tight', pad_inches=0.1)

    plt.show()
