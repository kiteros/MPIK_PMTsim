import numpy as np
from matplotlib import pyplot as plt
from shower_analysis import *

# --- start ---
if __name__ == "__main__":
    # inputs
    protons = "data/protonbbww/"
    gammas = "data/gammabbww/"

    # load edges
    xedges = np.load(protons+"xedges.npy")
    yedges = np.load(protons+"yedges.npy")

    # load events
    taggedPmtEvts = np.load(protons+"taggedPmtEvts2.npy")
    taggedPmtEvts = taggedPmtEvts[taggedPmtEvts["distance"] > 20*100]
    showerIdsP = np.unique(taggedPmtEvts["showerID"]).astype(int)

    # calculate likelihood ratio
    *_, histLR = makeHistograms(xedges, yedges, taggedPmtEvts)

    # select what to do
    edst = True
    if edst:
        plotEdst = True
        cuts = np.array([6,7.2,10.7])
        sep = np.linspace(0,200,200)
        eBinCnt = 4
    else:
        plotEdst = False
        cuts = np.linspace(5,40,50)
        sep = np.linspace(0,200,200)
        eBinCnt = 20


    # tag events
    cntsP, tCntsP = tagShowers(xedges, yedges, taggedPmtEvts, histLR, cut=cuts, truth=True)

    # repeat for gammas
    taggedPmtEvts = np.load(gammas+"taggedPmtEvts2.npy")
    taggedPmtEvts = taggedPmtEvts[taggedPmtEvts["distance"] > 20*100]
    cntsG, tCntsG = tagShowers(xedges, yedges, taggedPmtEvts, histLR, cut=cuts, truth=True)

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

    # energy distribution
    energyP = np.load(protons+"energy.npy")
    energyG = np.load(gammas+"energy.npy")
    energyP = energyP[showerIdsP]
    # plot
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
        snr = plotSeparationCuts(np.array(["{:.0f}@{:.0f}-{:.0f}TeV".format(c,minE/1000,maxE/1000) for c in cuts]),cntsPed,cntsGed,sep=sep,plot=plotEdst)
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



    plt.show()
