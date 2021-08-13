import numpy as np
from matplotlib import pyplot as plt

TAG_E = 1
TAG_MU = 2
TAG_MESON = 4
TAG_OTHER = 8
def makeTag(parts):
    tag = 0
    for p in parts:
        if p < 4: tag |= TAG_E
        elif p >= 5 and p < 7: tag |= TAG_MU
        elif p >= 7 and p < 13: tag |= TAG_MESON
        else: tag |= TAG_OTHER
    return tag

def makeHistograms(xedges, yedges, taggedPmtEvts):
    # extract relevant data
    upper = taggedPmtEvts[2]
    lower = taggedPmtEvts[3]
    tags = taggedPmtEvts[4].astype(int)
    tags |= taggedPmtEvts[5].astype(int)
    # select type
    eOnly = tags ^ TAG_E == 0
    #muOnly = tags ^ TAG_MU == 0
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
    paths = ["data/protonbbww/"]#, "data/gammabbww/"]
    for path in paths:
        xedges = np.load(path+"xedges.npy")
        yedges = np.load(path+"yedges.npy")
        histULAll = np.load(path+"histUL.npy")

        plotHists = False

        # fig 13 (upper/lower ratio)
        if plotHists: plotLogHist2d(xedges, yedges, histULAll, path)

        # make individual histograms
        taggedPmtEvts = np.load(path+"taggedPmtEvts.npy")
        xedges, yedges, upper, lower, muAny, histEOnly, histMuAny, histLR = makeHistograms(xedges, yedges, taggedPmtEvts)
        # plot
        for title, hist in zip(["Only electrons", "Any muons", "Likelihood ratio"],
                [histEOnly, histMuAny, histLR]):
            if plotHists: plotLogHist2d(xedges, yedges, hist, title)
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

