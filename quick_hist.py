import numpy as np
from matplotlib import pyplot as plt
from shower_analysis import *

# --- start ---
if __name__ == "__main__":
    # data
    paths = ["data/protonbbww/", "data/gammabbww/"]
    xedges = np.load(paths[0]+"xedges.npy")
    yedges = np.load(paths[0]+"yedges.npy")

    plotHists = True

    # fig 13 (upper/lower ratio)
    #if plotHists: plotLogHist2d(xedges, yedges, histULAll, path)

    # make individual histograms
    taggedPmtEvts, _ = loadData(paths,20)
    upper, lower, muAny, histEOnly, histMuAny, histLR = makeHistograms(xedges, yedges, taggedPmtEvts)
    histULAll, *_ = np.histogram2d(upper,lower,bins=[xedges,yedges])
    # plot
    for title, hist in zip(["Charge distribution histogram","Only electrons", "Any muons", "Likelihood ratio"],
            [histULAll, histEOnly, histMuAny, histLR]):
        if plotHists: plotLogHist2d(xedges, yedges, hist, title)
    # ROC curve
    muLR = muonScoreLR(xedges, yedges, upper, lower, histLR)
    cuts = np.linspace(0, 1)
    plotROC(muAny, muLR, cuts)

    # fig 14
    tagging = muLR > 10
    bins = np.logspace(0.5,4)
    tt = np.empty(bins.size-1)
    ff = np.empty(tt.size)
    tf = np.empty(tt.size)
    ft = np.empty(tt.size)
    for minPE, maxPE, i in zip(bins[:-1], bins[1:], np.arange(tt.size)):
        sel = np.logical_and(upper > minPE, upper < maxPE)
        tt[i] = np.logical_and(tagging[sel], muAny[sel]).sum()
        ff[i] = np.logical_and(~tagging[sel], ~muAny[sel]).sum()
        tf[i] = np.logical_and(~tagging[sel], muAny[sel]).sum()
        ft[i] = np.logical_and(tagging[sel], ~muAny[sel]).sum()
    plt.figure(14)
    plt.plot(bins[:-1],ff,color="orange",label="not mu & not tagged")
    plt.plot(bins[:-1],ft,color="orange",linestyle="--",label="not mu & tagged")
    plt.plot(bins[:-1],tt,color="blue",label="mu & tagged")
    plt.plot(bins[:-1],tf,color="blue",linestyle="--",label="mu & not tagged")
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("upper cell PEs")
    plt.ylabel("counts")



    plt.show()

