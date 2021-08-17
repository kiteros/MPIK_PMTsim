import numpy as np
from matplotlib import pyplot as plt
from shower_analysis import *
from mu_tag_LR import energyDependentAnalysis

def getT10_90(taggedPmtEvts):
    # get percentiles and differences
    firstUpper = taggedPmtEvts["firstUpper"]
    firstLower = taggedPmtEvts["firstLower"]
    per10Upper = taggedPmtEvts["per10Upper"] - firstUpper
    per10Lower = taggedPmtEvts["per10Lower"] - firstLower
    per90Upper = taggedPmtEvts["per90Upper"] - firstUpper
    per90Lower = taggedPmtEvts["per90Lower"] - firstLower
    diffUpper = per90Upper - per10Upper
    diffLower = per90Lower - per10Lower
    return diffUpper,diffLower

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
    showerIdsP = np.unique(taggedPmtEvts["showerID"]).astype(int)

    diffUpper, diffLower = getT10_90(taggedPmtEvts)

    # settings
    edst = True
    if edst:
        plotEdst = True
        cuts = np.array([1.6,2.6,3])
        sep = np.linspace(0,200,200)
        eBinCnt = 4
    else:
        plotEdst = False
        cuts = np.linspace(0,10,50)
        sep = np.linspace(0,100,100)
        eBinCnt = 20
    plotHists = False

    # histogram
    if plotHists:
        for q, l in zip([diffUpper, diffLower],#per10Upper, per10Lower, per90Upper, per90Lower,
            ["difference upper", "difference lower",]):#"0.1 upper", "0.1 lower","0.9 upper", "0.9 lower",
            hist, bins = np.histogram(q,range=(0,140))
            plt.figure()
            plt.title(l)
            plt.bar(bins[:-1],hist)

    # 2D histogram
    hist, xedges, yedges = np.histogram2d(diffUpper,diffLower,bins=(np.geomspace(0.01,500,101),np.geomspace(0.01,500,101)))
    if plotHists: plotLogHist2d(xedges,yedges,hist,xlabel="upper 10-90 time/ns",ylabel="lower 10-90 time/ns")

    # electron/muon histograms
    *_, muAny, histEOnly,histMuAny,histLR = makeHistograms(xedges,yedges,taggedPmtEvts,diffUpper,diffLower)
    for title, hist in zip(["Only electrons", "Any muons", "Likelihood ratio"],
            [histEOnly, histMuAny, histLR]):
        if plotHists: plotLogHist2d(xedges, yedges, hist, title, "upper 10-90 time/ns", "lower 10-90 time/ns")
    
    '''
    # muon tagging
    scores = muonScoreLR(xedges, yedges, diffUpper, diffLower, histLR)
    plotROC(muAny, scores, np.linspace(0,1))
    #'''

    # tag showers
    cntsP, tCntsP = tagShowers(xedges, yedges, taggedPmtEvts, histLR, cut=cuts, truth=True, upper=diffUpper, lower=diffLower)

    # repeat for gammas
    taggedPmtEvts = np.load(gammas+"taggedPmtEvts2.npy")
    taggedPmtEvts = taggedPmtEvts[taggedPmtEvts["distance"] > 20*100]
    diffUpper, diffLower = getT10_90(taggedPmtEvts)
    cntsG, tCntsG = tagShowers(xedges, yedges, taggedPmtEvts, histLR, cut=cuts, truth=True, upper=diffUpper, lower=diffLower)

    # analyse
    energyDependentAnalysis(protons, gammas, showerIdsP, plotEdst, cuts, sep, eBinCnt, cntsP, tCntsP, cntsG, tCntsG, False)

    plt.show()
