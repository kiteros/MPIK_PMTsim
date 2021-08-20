import numpy as np
from matplotlib import pyplot as plt
from shower_analysis import *
from mu_tag_LR import energyDependentAnalysis, plotRatioEMu, loadData
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
    taggedPmtEvtsP, showerIdsP, taggedPmtEvtsG, taggedPmtEvtsFull, energyP, energyG = loadData(protons, gammas, exclusion)

    diffUpper, diffLower = getT10_90(taggedPmtEvtsFull)

    # plot muon vs electron count
    eCntP, muCntP, eCntG, muCntG = plotRatioEMu(taggedPmtEvtsP, taggedPmtEvtsG, energyP, energyG)

    # settings
    edst = True
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
    *_, muAny, histEOnly,histMuAny,histLR = makeHistograms(xedges,yedges,taggedPmtEvtsFull,diffUpper,diffLower)
    for title, hist in zip(["Only electrons", "Any muons", "Likelihood ratio"],
            [histEOnly, histMuAny, histLR]):
        if plotHists: plotLogHist2d(xedges, yedges, hist, title, "upper 10-90 time/ns", "lower 10-90 time/ns")
    
    '''
    # muon tagging
    scores = muonScoreLR(xedges, yedges, diffUpper, diffLower, histLR)
    plotROC(muAny, scores, np.linspace(0,1))
    #'''

    # tag showers
    diffUpper, diffLower = getT10_90(taggedPmtEvtsFull)
    cntsP, tCntsP = tagShowers(xedges, yedges, taggedPmtEvtsP, histLR, cut=cuts, truth=True, upper=diffUpper, lower=diffLower, ratio=True)

    # repeat for gammas
    diffUpper, diffLower = getT10_90(taggedPmtEvtsG)
    cntsG, tCntsG = tagShowers(xedges, yedges, taggedPmtEvtsG, histLR, cut=cuts, truth=True, upper=diffUpper, lower=diffLower, ratio=True)

    # ratios
    plt.figure(2)
    plt.scatter(energyP/1000,cntsP[0],label="Protons (estimated)",marker=".")
    plt.scatter(energyG/1000,cntsG[0],label="Gammas (estimated)",marker="^")
    plt.legend()
    plt.figure()
    plt.title("Correlation")
    plt.scatter(muCntP/eCntP,cntsP[0],label="Protons",marker=".")
    plt.scatter(muCntG/eCntG,cntsG[0],label="Gammas",marker="^")
    plt.legend()

    # analyse
    energyDependentAnalysis(protons, gammas, showerIdsP, plotEdst, cuts, sep, eBinCnt, cntsP, tCntsP, cntsG, tCntsG, False)

    # make muon tagger
    MuTagRise(xedges,yedges,histLR).save("models/mu_tag_LR_rise")

    plt.show()
