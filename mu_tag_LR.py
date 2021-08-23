import numpy as np
from matplotlib import pyplot as plt
from shower_analysis import *
from muon_tagging import MuTagLR

# --- start ---
if __name__ == "__main__":
    # inputs
    protons = "data/protonbbww/"
    gammas = "data/gammabbww/"

    # load edges
    xedges = np.load(protons+"xedges.npy")
    yedges = np.load(protons+"yedges.npy")

    # load events
    exclusion = 0
    taggedPmtEvts, primaries = loadData([protons, gammas], exclusion)

    # calculate likelihood ratio
    *_, histLR = makeHistograms(xedges, yedges, taggedPmtEvts)

    # make muon tagger
    mt = MuTagLR(xedges,yedges,histLR)

    # plot muon vs electron count
    eCnt, muCnt = plotRatioEMu(taggedPmtEvts,primaries)

    # select what to do
    edst = True
    if edst:
        plotEdst = True
        cuts = np.array([10])#[6,7.2,10.7])
        sep = np.logspace(-10,0,200) #np.linspace(0,200,200)
        eBinCnt = 4
    else:
        plotEdst = False
        cuts = np.linspace(0,40,50)
        sep = np.logspace(-10,0,200) #np.linspace(0,200,200)
        eBinCnt = 20


    # tag events
    cnts, tCnts, ids = tagShowers(mt, taggedPmtEvts, cut=cuts, truth=True, ratio=True, makeIds=True)
    selP = primaries["showerType"][ids] == ID_PROTON
    selG = primaries["showerType"][ids] == ID_PHOTON
    

    # plot counts/ratios again
    '''
    plt.figure(1)
    eCntsP = eCntP+muCntP-cntsP[0]
    plt.scatter(cntsP[0],eCntsP,label="Protons (estimated)",marker=".")
    eCntsG = eCntG+muCntG-cntsG[0]
    plt.scatter(cntsG[0],eCntsG,label="Gammas (estimated)",marker="^")
    plt.legend()#'''
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
    energyDependentAnalysis(cnts, tCnts, cuts, sep, ids, primaries, plotEdst, eBinCnt)

    # save muon tagger
    mt.save("models/mu_tag_LR_PEs")

    plt.show()
