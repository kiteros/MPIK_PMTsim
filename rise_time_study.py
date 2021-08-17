import numpy as np
from matplotlib import pyplot as plt
from shower_analysis import makeHistograms, muonScoreLR, plotLogHist2d, TAG_E, TAG_MU, plotROC

# --- start ---
if __name__ == "__main__":
    protons = "data/protonbbww/"
    gammas = "data/gammabbww/"

    # load events
    taggedPmtEvts = np.load(protons+"taggedPmtEvts2.npy")
    taggedPmtEvts = taggedPmtEvts[taggedPmtEvts["distance"] > 20*100]
    # for meaningful analysis, more than two PEs
    #taggedPmtEvts = taggedPmtEvts[taggedPmtEvts["lower"] > 2]
    #taggedPmtEvts = taggedPmtEvts[taggedPmtEvts["upper"] > 2]

    # get percentiles and differences
    firstUpper = taggedPmtEvts["firstUpper"]
    firstLower = taggedPmtEvts["firstLower"]
    per10Upper = taggedPmtEvts["per10Upper"] - firstUpper
    per10Lower = taggedPmtEvts["per10Lower"] - firstLower
    per90Upper = taggedPmtEvts["per90Upper"] - firstUpper
    per90Lower = taggedPmtEvts["per90Lower"] - firstLower
    diffUpper = per90Upper - per10Upper
    diffLower = per90Lower - per10Lower

    # histogram
    for q, l in zip([diffUpper, diffLower],#per10Upper, per10Lower, per90Upper, per90Lower,
        ["difference upper", "difference lower",]):#"0.1 upper", "0.1 lower","0.9 upper", "0.9 lower",
        hist, bins = np.histogram(q,range=(0,140))
        plt.figure()
        plt.title(l)
        plt.bar(bins[:-1],hist)

    # 2D histogram
    hist, xedges, yedges = np.histogram2d(diffUpper,diffLower,bins=(np.geomspace(0.01,500,101),np.geomspace(0.01,500,101)))
    plotLogHist2d(xedges,yedges,hist,xlabel="upper 10-90 time/ns",ylabel="lower 10-90 time/ns")

    # electron/muon histograms
    *_, muAny, histEOnly,histMuAny,histLR = makeHistograms(xedges,yedges,taggedPmtEvts,diffUpper,diffLower)
    # plot
    plotHists = True
    for title, hist in zip(["Only electrons", "Any muons", "Likelihood ratio"],
            [histEOnly, histMuAny, histLR]):
        if plotHists: plotLogHist2d(xedges, yedges, hist, title, "upper 10-90 time/ns", "lower 10-90 time/ns")
    
    # muon tagging
    scores = muonScoreLR(xedges, yedges, diffUpper, diffLower, histLR)
    plotROC(muAny, scores, np.linspace(0,1))

    plt.show()
