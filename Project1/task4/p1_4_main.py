import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm as lpnorm

if __name__ == "__main__":
    N = 1000    # Precision
    p = 0.5     # p-norm

    # Discretize unit-circle
    angles = np.linspace(0, 2*np.pi, N)
    # Create unit-circle points
    points = np.stack((np.cos(angles), np.sin(angles)), 1)

    # Normalize them with p-norm
    points = (points.T / np.array([lpnorm(point, p) for point in points])).T

    # Plot
    plt.plot(points[:, 0], points[:, 1], linestyle='-')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().set_title('Unit Circle: p = ' + str(p))
    plt.show()