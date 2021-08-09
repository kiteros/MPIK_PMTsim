import numpy as np
from matplotlib import pyplot as plt

# data
paths = ["data/protonbbww/", "data/gammabbww/"]
for path in paths:
    xedges = np.load(path+"xedges.npy")
    yedges = np.load(path+"yedges.npy")
    histULAll = np.load(path+"histUL.npy")

    # fig 13 (upper/lower ratio)
    plt.figure()
    plt.title(path)
    plt.pcolormesh(*np.meshgrid(xedges, yedges), np.log(histULAll))
    plt.colorbar()
    plt.xscale("log")
    plt.yscale("log")


plt.show()

