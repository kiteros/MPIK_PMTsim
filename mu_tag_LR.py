from re import X
import numpy as np
from matplotlib import pyplot as plt
from shower_analysis import *
from muon_tagging import MuTagLR

def energyDependentAnalysis(protons, gammas, showerIdsP, plotEdst, cuts, sep, eBinCnt, cntsP, tCntsP, cntsG, tCntsG, plotProfiles=True):
    # energy distribution
    energyP = np.load(protons+"energy.npy")
    energyG = np.load(gammas+"energy.npy")
    energyP = energyP[showerIdsP]
    # plot
    if plotProfiles:
        plt.figure(4)
        plt.errorbar(*profilePoints(energyG, tCntsG), label="gammas")
        plt.errorbar(*profilePoints(energyP, tCntsP), label="protons")
        plt.xlabel("Energy")
        plt.ylabel("Muons")

    # energy dependent cuts
    if plotEdst:
        plt.figure(5)
        plt.title("Energy dependent shower tagging")
    energyBins = np.histogram_bin_edges(energyP,eBinCnt)
    snrBest = np.empty((energyBins.size-1,5))
    for minE, maxE, i in zip(energyBins[:-1],energyBins[1:], np.arange(snrBest.size)):
        sel = np.logical_and(energyP > minE, energyP < maxE)
        cntsPed = cntsP[:,sel]
        sel = np.logical_and(energyG > minE, energyG < maxE)
        cntsGed = cntsG[:,sel]
        snr = plotSeparationCuts(np.array(["{:.1f}@{:.0f}-{:.0f}TeV".format(c,minE/1000,maxE/1000) for c in cuts]),cntsPed,cntsGed,sep=sep,plot=plotEdst)
        amax = np.unravel_index(np.nanargmax(snr), snr.shape)
        snrBest[i] = np.array([amax[0], amax[1], np.nanmax(snr), minE, maxE])
    if plotEdst:
        plt.colorbar(label="minimum muons per shower")

    # plot signal to background ratio
    plt.figure(6)
    plt.title("Signal to background ratio")
    plt.scatter(cuts[snrBest[:,0].astype(int)],sep[snrBest[:,1].astype(int)],c=np.log(snrBest[:,2]))
    for x,y,snr,minE,maxE in snrBest:
        plt.annotate("{:.0f}-{:.0f}TeV".format(minE/1000,maxE/1000),(cuts[int(x)],sep[int(y)]))
    plt.colorbar(label="log(signal to background ratio)")
    plt.xlabel("LR cut")
    plt.ylabel("Muon cut")

    # plot muon estimates
    '''
    plt.figure()
    plt.title("Muon counts")
    for x,y,snr,minE,maxE in snrBest:
        sel = np.logical_and(energyP > minE, energyP < maxE)
        plt.scatter(tCntsP[sel], cntsP[int(x)][sel]-tCntsP[sel], label="{:.1f}@{:.0f}-{:.0f}TeV".format(cuts[int(x)],minE/1000,maxE/1000),marker=".")
        sel = np.logical_and(energyG > minE, energyG < maxE)
        plt.scatter(tCntsG[sel], cntsG[int(x)][sel]-tCntsG[sel], label="{:.1f}@{:.0f}-{:.0f}TeV".format(cuts[int(x)],minE/1000,maxE/1000),marker="^")
    plt.plot([0,tCntsP.max()],[0,0])
    plt.xlabel("True muon number")
    plt.ylabel("Estimated muons number")
    plt.legend()#'''

def plotRatioEMu(taggedPmtEvtsP, taggedPmtEvtsG, energyP, energyG):
    eOnlyP, muAnyP = getEMuTags(taggedPmtEvtsP)
    cdx = taggedPmtEvtsP["showerID"]
    eCntP = magicCumsum(cdx,eOnlyP)
    muCntP = magicCumsum(cdx,muAnyP)
    eOnlyG, muAnyG = getEMuTags(taggedPmtEvtsG)
    cdx = taggedPmtEvtsG["showerID"]
    eCntG = magicCumsum(cdx,eOnlyG)
    muCntG = magicCumsum(cdx,muAnyG)
    plt.figure(1)
    plt.title("Electron muon ratio")
    plt.scatter(muCntP,eCntP,label="Protons",marker=".")
    plt.scatter(muCntG,eCntG,label="Gammas",marker="^")
    plt.xlabel("Muon events")
    plt.ylabel("Electron only events")
    plt.legend()
    # plot muon electron ratio
    plt.figure(2)
    plt.title("Electron muon ratio")
    plt.scatter(energyP/1000,muCntP/eCntP,label="Protons",marker=".")
    plt.scatter(energyG/1000,muCntG/eCntG,label="Gammas",marker="^")
    plt.xlabel("Energy/TeV")
    plt.ylabel("$N_\mu/N_e$")
    plt.legend()
    return eCntP,muCntP,eCntG,muCntG

def loadData(protons, gammas, exclusion):
    taggedPmtEvtsP = np.load(protons+"taggedPmtEvts2.npy")
    taggedPmtEvtsP = taggedPmtEvtsP[taggedPmtEvtsP["distance"] > exclusion*100]
    showerIdsP = np.unique(taggedPmtEvtsP["showerID"]).astype(int)
    taggedPmtEvtsG = np.load(gammas+"taggedPmtEvts2.npy")
    taggedPmtEvtsG = taggedPmtEvtsG[taggedPmtEvtsG["distance"] > exclusion*100]
    showerIdsG = np.unique(taggedPmtEvtsG["showerID"]).astype(int)

    #TODO ensure valid shower ids and match with primaries, best case: make them usable as index (primaries["showerID"][i] == i)
    taggedPmtEvtsFull = np.concatenate((taggedPmtEvtsG, taggedPmtEvtsP))

    # load energies
    energyP = np.load(protons+"energy.npy")
    energyG = np.load(gammas+"energy.npy")
    energyP = energyP[showerIdsP]
    return taggedPmtEvtsP,showerIdsP,taggedPmtEvtsG,taggedPmtEvtsFull,energyP,energyG

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
    taggedPmtEvtsP, showerIdsP, taggedPmtEvtsG, taggedPmtEvtsFull, energyP, energyG = loadData(protons, gammas, exclusion)

    # calculate likelihood ratio
    *_, histLR = makeHistograms(xedges, yedges, taggedPmtEvtsFull)

    # plot muon vs electron count
    eCntP, muCntP, eCntG, muCntG = plotRatioEMu(taggedPmtEvtsP, taggedPmtEvtsG, energyP, energyG)

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
    cntsP, tCntsP = tagShowers(xedges, yedges, taggedPmtEvtsP, histLR, cut=cuts, truth=True, ratio=True)

    # repeat for gammas
    cntsG, tCntsG = tagShowers(xedges, yedges, taggedPmtEvtsG, histLR, cut=cuts, truth=True, ratio=True)

    # plot counts/ratios again
    '''
    plt.figure(1)
    eCntsP = eCntP+muCntP-cntsP[0]
    plt.scatter(cntsP[0],eCntsP,label="Protons (estimated)",marker=".")
    eCntsG = eCntG+muCntG-cntsG[0]
    plt.scatter(cntsG[0],eCntsG,label="Gammas (estimated)",marker="^")
    plt.legend()#'''
    plt.figure(2)
    plt.scatter(energyP/1000,cntsP[0],label="Protons (estimated)",marker=".")
    plt.scatter(energyG/1000,cntsG[0],label="Gammas (estimated)",marker="^")
    plt.legend()
    plt.figure()
    plt.title("Correlation")
    plt.scatter(muCntP/eCntP,cntsP[0],label="Protons",marker=".")
    plt.scatter(muCntG/eCntG,cntsG[0],label="Gammas",marker="^")
    plt.legend()

    # plot
    '''
    #plt.figure(1)
    #plt.hist(sums, label="protons")
    plt.figure(2)
    bins = np.linspace(0,100)
    colors = plt.get_cmap("tab20")
    for i in np.arange(cuts.size):
        plt.hist(cntsP[i], bins=bins, histtype="step", color=colors(i*2), label="protons")
        plt.hist(tCntsP[i], bins=bins, histtype="step", linestyle="--", color=colors(i*2), label="protons(true)")
        plt.hist(cntsG[i], bins=bins, histtype="step", color=colors(i*2+1), label="gammas")
        plt.hist(tCntsG[i], bins=bins, histtype="step", linestyle="--", color=colors(i*2+1), label="gammas(true)")
    #plt.hist(cntsP-tCntsP, bins=bins, histtype="step", linestyle=":", label="protons(mistag)")
    #plt.hist(cntsG-tCntsG, bins=bins, histtype="step", linestyle=":", label="gammas(mistag)")
    plt.legend()#'''

    # find separation
    '''
    plt.figure(3)
    plt.title("Shower tagging")
    plotSeparationCuts(cuts, cntsP, cntsG)
    plt.colorbar(label="minimum muons per shower")
    #'''

    energyDependentAnalysis(protons, gammas, showerIdsP, plotEdst, cuts, sep, eBinCnt, cntsP, tCntsP, cntsG, tCntsG)

    # make muon tagger
    MuTagLR(xedges,yedges,histLR).save("models/mu_tag_LR_PEs")

    plt.show()
