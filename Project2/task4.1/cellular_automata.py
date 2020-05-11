import math
import numpy as np
import scipy.misc as msc
import scipy.ndimage as img
from PIL import Image
import matplotlib.pyplot as plt


def calculate_product_vector(initial_vector):
    result = np.array([1])
    for i in initial_vector:
        second_half = i * result
        result = np.append(result, second_half)
    return result


if __name__ == "__main__":
    rule110 = np.array([-1, 1, 1, -1, 1, 1, 1, -1])
    rule126 = np.array([-1, 1, 1, 1, 1, 1, 1, -1])

    X = np.array([[-1, -1, -1], [-1, -1,  1], [-1,  1, -1], [-1,  1,  1],
                  [ 1, -1, -1], [ 1, -1,  1], [ 1,  1, -1], [ 1,  1,  1]])

    w110 = np.linalg.inv(X.T @ X) @ X.T @ rule110
    y110 = X @ w110
    #print(y110)

    w126 = np.linalg.inv(X.T @ X) @ X.T @ rule126
    y126 = X @ w126
    #print(y126)


    F = np.apply_along_axis(calculate_product_vector, 1, X)

    w110_new = np.linalg.inv(F.T @ F) @ F.T @ rule110
    y110_new = F @ w110_new
    print(y110_new)

    w126_new = np.linalg.inv(F.T @ F) @ F.T @ rule126
    y126_new = F @ w126_new
    print(y126_new)
