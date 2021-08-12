import numpy as np
from matplotlib import pyplot as plt

#TODO import
TAG_E = 1
TAG_MU = 2
TAG_MESON = 4
TAG_OTHER = 8

def makeHistograms(xedges, yedges, taggedPmtEvts):
    # extract relevant data
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
    #histMuOnly, xedges, yedges = np.histogram2d(upper[muOnly],lower[muOnly], bins=[xedges,yedges])
    histMuAny, xedges, yedges = np.histogram2d(upper[muAny],lower[muAny], bins=[xedges,yedges])
    # scale hists
    histEOnly /= np.sum(histEOnly)
    histMuAny /= np.sum(histMuAny)
    # likelihood ratio
    histLR = histMuAny/histEOnly
    return xedges,yedges,upper,lower,muAny,histEOnly,histMuAny,histLR

def plotLogHist2d(xedges, yedges, hist, title=None, xlabel="upper cell PEs", ylabel="lower cell PEs"):
    plt.figure()
    plt.title(title)
    plt.pcolormesh(*np.meshgrid(xedges, yedges), np.log(hist))
    plt.colorbar()
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

def muonScoreLR(xedges, yedges, upper, lower, histLR):
    uppIdx = np.digitize(upper, xedges)
    lowIdx = np.digitize(lower, yedges)
    uppIdx[uppIdx >= xedges.shape[0]] = xedges.shape[0]-2
    lowIdx[lowIdx >= yedges.shape[0]] = yedges.shape[0]-2
    muLR = histLR[uppIdx, lowIdx]
    return muLR

# --- start ---
if __name__ == "__main__":
    # data
    paths = ["data/protonbbww/", "data/gammabbww/"]
    for path in paths:
        xedges = np.load(path+"xedges.npy")
        yedges = np.load(path+"yedges.npy")
        histULAll = np.load(path+"histUL.npy")

        # fig 13 (upper/lower ratio)
        plotLogHist2d(xedges, yedges, histULAll, path)

        # make individual histograms
        taggedPmtEvts = np.load(path+"taggedPmtEvts.npy")
        xedges, yedges, upper, lower, muAny, histEOnly, histMuAny, histLR = makeHistograms(xedges, yedges, taggedPmtEvts)
        # plot
        for title, hist in zip(["Only electrons", "Any muons", "Likelihood ratio"],
                [histEOnly, histMuAny, histLR]):
            plotLogHist2d(xedges, yedges, hist, title)
        # ROC curve
        muLR = muonScoreLR(xedges, yedges, upper, lower, histLR)
        cuts = np.linspace(0, 1)
        falseMu = np.zeros(cuts.shape)
        trueMu = np.zeros(cuts.shape)
        for i in np.arange(cuts.shape[0]):
            tagging = muLR > cuts[i]
            trueMu[i] = np.logical_and(tagging,muAny).sum()/muAny.sum()
            falseMu[i] = np.logical_and(tagging,~muAny).sum()/(~muAny).sum()
        # plot
        plt.figure()
        plt.title("ROC curve")
        plt.plot(falseMu,trueMu)
        plt.scatter(falseMu,trueMu,c=cuts)
        plt.colorbar()
        plt.xlabel("false muons")
        plt.ylabel("true muons")




    plt.show()

