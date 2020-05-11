import math
import numpy as np
import scipy.misc as msc
import scipy.ndimage as img


def foreground2_bin_img(f):

    d = img.filters.gaussian_filter(f, sigma=0.50, mode="reflect") - \
    img.filters.gaussian_filter(f, sigma=1.00, mode="reflect")
    d = np.abs(d)
    m = d.max()
    d[d< 0.1*m] = 0
    d[d>=0.1*m] = 1
    return img.morphology.binary_closing(d)


def calc_fractal_dim(g):
    """
    Calculates Fractal Dimension of Binarized Image g

    :param g: Binarized ndarray representing Image matrix
    :return D: fractal dimension of g.
    """
    w, h = g.shape
    L = int(math.log2(w))

    # Populating set of Scaling Factors
    S = [1 / 2 ** i for i in range(1, L - 2)]

    # checks if a block has non-zero elements
    chk = lambda x_block: 1 if np.sum(x_block)>1 else 0

    # provides a list of non-overlapping blocks of side length (s_i*h, s_i*w) from g
    blocks = lambda s_i: [g[i:min(h,i+int(s_i*h)), j:min(w,j+int(s_i*w))] for i in range(0,h,int(s_i*h))\
                                                                        for j in range(0,w,int(s_i*w))]

    # calculating the number of blocks with non-zero elements for all scaling factors in s_i
    n = [sum([chk(b) for b in blocks(s_i)]) for s_i in S]

    # Mapping n and S to x-y values for final line-fitting
    y = list(map(math.log,n))
    x = list(map(lambda s_i:math.log(1/s_i), S))

    # Fitting a degree 1 polynomial to x and y
    p = np.polyfit(x, y, deg=1) # Highest power first
    #print(p)
    return p[0]


if __name__ == "__main__":
    # Fractal Dimension calculations
    imgName = "lightning-3"
    f = msc.imread(imgName+".png", flatten=True).astype(np.float)
    g = foreground2_bin_img(f)
    g = g*1
    print("Fractal Dimension for {}.png = {}".format(imgName, calc_fractal_dim(g)))

    imgName = "tree-2"
    f = msc.imread(imgName + ".png", flatten=True).astype(np.float)
    g = foreground2_bin_img(f)
    g = g * 1
    print("Fractal Dimension for {}.png = {}".format(imgName, calc_fractal_dim(g)))