import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # read data as 2D array of data type 'object'
    data = np.loadtxt('whData.dat', dtype=np.object, comments='#', delimiter=None)

    # read height data into 2D array (i.e. into a matrix)
    X = data[:, 1].astype(np.float)

    # Sample mean and std enough to parameterize a normal distribution
    mean = np.mean(X)
    std = np.std(X)

    # Plot the height data as circle markers
    plt.plot(X, np.zeros_like(X) + 0, 'o', color='b', alpha=0.5, label='data')

    # Plot the Gaussian fit for height data
    xmin, xmax = 140, 210
    x = np.linspace(xmin, xmax, 1000) # 1000 evenly spaced numbers over xmin and xmax for plotting normal distribution

    p = norm.pdf(x, mean, std) # specify a normal distribution for sample mean and std
    plt.plot(x, p, 'k', linewidth=2, color='y', label='normal') # Plot the gaussian

    plt.ylim(0,0.06)
    #specifying title and plotting title and legend
    title = "Univariate Gaussian Distribution fit to: mu = %.2f,  std = %.2f" % (mean, std)
    plt.title(title)
    plt.legend()

    #Saving the figure
    plt.savefig("gaussian_fit_to_height.pdf",  # facecolor='w', edgecolor='w',
                papertype=None, format='pdf', transparent=False,
                bbox_inches='tight', pad_inches=0.1)

    # Display figure
    plt.show()