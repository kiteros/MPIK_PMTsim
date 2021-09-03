import numpy as np
from matplotlib import pyplot as plt
from shower_analysis import *
from muon_tagging import MuTagRise

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
    exclusion = 20
    taggedPmtEvts, primaries = loadData([protons, gammas], exclusion)

    p = "50"
    diffUpper = taggedPmtEvts["per"+p+"Upper"]-taggedPmtEvts["per10Upper"]
    diffLower = taggedPmtEvts["per"+p+"Lower"]-taggedPmtEvts["per10Lower"]

    # plot muon vs electron count
    eCnt, muCnt = plotRatioEMu(taggedPmtEvts,primaries)

    # settings
    edst = False
    if edst:
        plotEdst = True
        cuts = np.array([2])
        sep =  np.logspace(-10,0,200) #np.linspace(0,200,200)
        eBinCnt = 4
    else:
        plotEdst = False
        cuts = np.linspace(0,10,50)
        sep = np.logspace(-10,0,200) #np.linspace(0,100,100)
        eBinCnt = 20
    plotHists = True

    # histogram
    if plotHists:
        for q, l in zip([diffUpper, diffLower],#per10Upper, per10Lower, per90Upper, per90Lower,
            ["difference upper", "difference lower",]):#"0.1 upper", "0.1 lower","0.9 upper", "0.9 lower",
            hist, bins = np.histogram(q,range=(0,140))
            plt.figure()
            plt.title(l)
            plt.bar(bins[:-1],hist)

    # 2D histogram
    bins = np.geomspace(0.01,500,101)
    hist, xedges, yedges = np.histogram2d(np.clip(diffUpper,bins[0],bins[-1]),np.clip(diffLower,bins[0],bins[-1]),bins=(bins,bins))
    if plotHists: plotLogHist2d(xedges,yedges,hist,title="Rise time histogram",xlabel="upper 10-"+p+" time/ns",ylabel="lower 10-"+p+" time/ns")

    # electron/muon histograms
    *_, muAny, histEOnly,histMuAny,histLR = makeHistograms(xedges,yedges,taggedPmtEvts,np.clip(diffUpper,bins[0],bins[-1]),np.clip(diffLower,bins[0],bins[-1]),clip=False)
    for title, hist in zip(["Only electrons", "Any muons", "Likelihood ratio"],
            [histEOnly, histMuAny, histLR]):
        if plotHists: plotLogHist2d(xedges, yedges, hist, title, "upper 10-"+p+" time/ns", "lower 10-"+p+" time/ns")
    
    # make muon tagger
    mt = MuTagRise(xedges,yedges,histLR,{"high":p,"low":"10"})

    # tag events
    cnts, tCnts, ids = tagShowers(mt, taggedPmtEvts, cut=cuts, truth=True, ratio=True, makeIds=True)
    selP = primaries["showerType"][ids] == ID_PROTON
    selG = primaries["showerType"][ids] == ID_PHOTON
    

    # plot counts/ratios again
    plt.figure(2)
    plt.scatter(primaries["showerEnergy"][ids][selP]/1000,cnts[0][selP],label="Protons (estimated)",marker=".")
    plt.scatter(primaries["showerEnergy"][ids][selG]/1000,cnts[0][selG],label="Gammas (estimated)",marker="^")
    plt.legend()
    plt.figure()
    plt.title("Correlation")
    plt.scatter(muCnt[selP]/eCnt[selP],cnts[0][selP],label="Protons",marker=".")
    plt.scatter(muCnt[selG]/eCnt[selG],cnts[0][selG],label="Gammas",marker="^")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()

    # analyse wrt energy
    #energyDependentAnalysis(cnts, tCnts, cuts, sep, ids, primaries, plotEdst, eBinCnt)

    # make muon tagger
    mt.save("models/mu_tag_LR_rise")

    plt.show()
