import numpy as np
from matplotlib import pyplot as plt
from quick_hist import makeHistograms, muonScoreLR, TAG_MU


def tagShowers(xedges, yedges, taggedPmtEvts, histLR, cut=1, truth=False):
    upper = taggedPmtEvts[2]
    lower = taggedPmtEvts[3]
    # calculate muon score
    score = muonScoreLR(xedges, yedges, upper, lower, histLR)
    # get shower indices
    cdx = taggedPmtEvts[0]
    indices = np.nonzero(np.r_[1, np.diff(cdx)[:-1]])[0]
    idx = np.empty(indices.shape, dtype=int)
    idx[:-1] = indices[1:]-1
    idx[-1] = cdx.size-1
    # sum muons
    if not isinstance(cut, np.ndarray):
        cuts = np.array([cut])
    else:
        cuts = cut
    cnts = np.empty((cuts.size, idx.size))
    for i in np.arange(cuts.size):
        cumsums = np.cumsum(score > cuts[i])[idx]
        cnts[i,1:] = cumsums[1:] - cumsums[0:-1]
        cnts[i,0] = cumsums[0]
    if not isinstance(cut, np.ndarray):
        cnts = cnts[0]
    #TODO handle inf/nan in sums
    if not truth: return cnts
    # sum true muons
    tags = taggedPmtEvts[4].astype(int)
    tags |= taggedPmtEvts[5].astype(int)
    muAny = tags & TAG_MU > 0
    cumsums = np.cumsum(muAny)[idx]
    tCnts = np.empty(cumsums.shape)
    tCnts[1:] = cumsums[1:] - cumsums[0:-1]
    tCnts[0] = cumsums[0]
    return cnts, tCnts


def plotSeparationCuts(cuts, cntsP, cntsG, sep=None, plot=True):
    if not isinstance(sep,np.ndarray):
        sep = np.linspace(0,200)
    sepG = np.empty(sep.shape)
    sepP = np.empty(sep.shape)
    snr = np.empty((cuts.size, sep.size))
    for cp, cg, c, j in zip(cntsP, cntsG, cuts, np.arange(cuts.size)):
        for i in np.arange(sep.size):
            sepG[i] = (cg > sep[i]).sum()
            sepP[i] = (cp > sep[i]).sum()
        sepG /= cg.shape[0]
        sepP /= cp.shape[0]
        snr[j,:] = (1-sepG)/(1-sepP+0.000000001)
        # plot
        if plot:
            plt.plot(sepG, sepP, label="cut="+str(c))
            plt.xlabel("tagged gammas")
            plt.ylabel("tagged protons")
            plt.scatter(sepG,sepP,c=sep)
            plt.legend()
    return snr


def profilePoints(xs, ys):
    bins = np.histogram_bin_edges(xs)
    x = np.empty(bins.size-1)
    xerr = np.empty(x.shape)
    y = np.empty(x.shape)
    yerr = np.empty(x.shape)
    for i in np.arange(x.size):
        sel = np.logical_and(xs > bins[i],xs <= bins[i+1])
        x[i] = np.mean(xs[sel])
        xerr[i] = np.std(xs[sel])
        y[i] = np.mean(ys[sel])
        yerr[i] = np.std(ys[sel])
    return x,y,yerr,xerr


# --- start ---
if __name__ == "__main__":
    # inputs
    protons = "data/protonbbww/"
    gammas = "data/gammabbww/"

    # load edges
    xedges = np.load(protons+"xedges.npy")
    yedges = np.load(protons+"yedges.npy")

    # load events
    taggedPmtEvts = np.load(protons+"taggedPmtEvts.npy")
    showerIdsP = np.unique(taggedPmtEvts[0]).astype(int)

    # calculate likelihood ratio
    xedges, yedges, upper, lower, *_, histLR = makeHistograms(xedges, yedges, taggedPmtEvts)

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
    taggedPmtEvts = np.load(gammas+"taggedPmtEvts.npy")
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
