import numpy as np
from matplotlib import pyplot as plt

#TODO import
TAG_E = 1
TAG_MU = 2
TAG_MESON = 4
TAG_OTHER = 8

# data
paths = ["data/protonbbww/"]#, "data/gammabbww/"]
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

    # make individual histograms
    taggedPmtEvts = np.load(path+"taggedPmtEvts.npy")
    upper = taggedPmtEvts[2]
    lower = taggedPmtEvts[3]
    tags = taggedPmtEvts[4].astype(int)
    tags |= taggedPmtEvts[5].astype(int)
    # select type
    eOnly = tags ^ TAG_E == 0
    muOnly = tags ^ TAG_MU == 0
    muAny = tags & TAG_MU > 0
    # histogram
    histEOnly, xedges, yedges = np.histogram2d(upper[eOnly],lower[eOnly], bins=[xedges,yedges])
    histMuOnly, xedges, yedges = np.histogram2d(upper[muOnly],lower[muOnly], bins=[xedges,yedges])
    histMuAny, xedges, yedges = np.histogram2d(upper[muAny],lower[muAny], bins=[xedges,yedges])
    # plot
    for title, hist in zip(["Only electrons", "Only muons", "Any muons"], [histEOnly, histMuOnly, histMuAny]):
        plt.figure()
        plt.title(title)
        plt.pcolormesh(*np.meshgrid(xedges, yedges), np.log(hist))
        plt.colorbar()
        plt.xscale("log")
        plt.yscale("log")



plt.show()

