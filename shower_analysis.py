import numpy as np
from matplotlib import pyplot as plt

TYPE_TAGGED_PMT_EVTS = [("showerID","i4"),("pmtIdG","i4"),("upper","i4"),("lower","i4"),("tagsUpper","int8"),("tagsLower","int8"),("distance","f4"),
    ("firstUpper","f4"),("firstLower","f4"),("per10Upper","f4"),("per10Lower","f4"),("per20Upper","f4"),("per20Lower","f4"),("per50Upper","f4"),("per50Lower","f4"),
    ("per90Upper","f4"),("per90Lower","f4")]
"""
Data type of `taggedPmtEvents` as used in all functions in this file.
"""

TYPE_PRIMARIES = [("showerID","i4"),("showerType","i4"),("showerEnergy","f4")]
"""
Data type of `primaries` as used in all functions in this file.
"""

ID_PHOTON = 1
ID_PROTON = 14

TAG_E = 1
TAG_MU = 2
TAG_MESON = 4
TAG_OTHER = 8
def makeTag(parts):
    """
    Calculates particle tags from list of particle IDs.

    Parameters
    ----------
    parts : array_like
        list of particle IDs
    
    Returns
    -------
    int
        particle tag (bitwise or of TAG_* values)
    """
    tag = 0
    for p in parts:
        if p < 4: tag |= TAG_E
        elif p >= 5 and p < 7: tag |= TAG_MU
        elif p >= 7 and p < 13: tag |= TAG_MESON
        else: tag |= TAG_OTHER
    return tag

def loadData(paths, exclusion=0):
    """
    Loads data from a list of paths.

    Parameters
    ----------
    paths : array_like
        list of path names
    exclusion : float, optional
        size of exclusion area in m, default is 0
    
    Returns
    -------
    taggedPmtEvts : structured array
        list of PMT events with tags (see `TYPE_TAGGED_PMT_EVTS`)
    primaries : structured array
        list of primary particles (see `TYPE_PRIMARIES`)
    """
    # load files
    tpes = []
    primaries = []
    offset = 0
    for path in paths:
        tpe = np.load(path+"taggedPmtEvts2.npy")
        tpe = tpe[tpe["distance"] > exclusion*100]
        tpe["showerID"] += offset
        tpes.append(tpe)
        pri = np.load(path+"primaries.npy")
        pri["showerID"] += offset
        primaries.append(pri)
        offset += pri.shape[0]

    # concatenate
    taggedPmtEvtsFull = np.concatenate(tpes)
    primaries = np.concatenate(primaries)
    #TODO ensure even for bad showerIDs
    assert np.all([primaries["showerID"][i] == i for i in np.arange(primaries.shape[0])])

    return taggedPmtEvtsFull,primaries

def makeHistograms(xedges, yedges, taggedPmtEvts, upper="upper", lower="lower"):
    """
    Makes histograms from tagged PMT events.

    Parameters
    ----------
    xedges : array_like
        histogram xedges
    yedges : array_like
        histogram yedges
    taggedPmtEvts : structured array
        list of PMT events with tags (see `TYPE_TAGGED_PMT_EVTS`)
    upper : string or array_like
        upper events, default is `"upper"`
    lower : string or array_like
        lower events, default is `"lower"`
    
    Returns
    -------
    upper : ndarray
        events in upper chamber
    lower : ndarray
        events in lower chamber
    muAny : ndarray
        if an muon is in the chamber
    histEOnly : ndarray
        histogram for electron only events
    histMuAny : ndarray
        histogram for muon events
    histLR : ndarray
        likelihood ratio of a muon event
    """
    # extract relevant data
    if isinstance(upper,str): upper = taggedPmtEvts[upper]
    if isinstance(lower,str): lower = taggedPmtEvts[lower]
    eOnly, muAny = getEMuTags(taggedPmtEvts)
    # histogram
    histEOnly, *_ = np.histogram2d(upper[eOnly],lower[eOnly], bins=[xedges,yedges])
    histMuAny, *_ = np.histogram2d(upper[muAny],lower[muAny], bins=[xedges,yedges])
    # scale hists
    histEOnly /= np.sum(histEOnly)
    histMuAny /= np.sum(histMuAny)
    # likelihood ratio
    histLR = histMuAny/histEOnly
    return upper,lower,muAny,histEOnly,histMuAny,histLR

def getEMuTags(taggedPmtEvts):
    """
    Creates bool arrays for electron and muon events.

    Parameters
    ----------
    taggedPmtEvts : structured array
        list of PMT events with tags (see `TYPE_TAGGED_PMT_EVTS`)
    
    Returns
    -------
    eOnly : ndarray
        bool array for electron only events
    muAny : ndarray
        bool array for events with muons
    """
    tags = taggedPmtEvts["tagsUpper"]
    tags |= taggedPmtEvts["tagsLower"]
    eOnly = tags ^ TAG_E == 0
    muAny = tags & TAG_MU > 0
    return eOnly, muAny

def plotLogHist2d(xedges, yedges, hist, title=None, xlabel="upper cell PEs", ylabel="lower cell PEs"):
    """
    Plots logarithmic 2D histogram.

    Parameters
    ----------
    xedges : array_like
        histogram xedges
    yedges : array_like
        histogram yedges
    hist : array_like
        particle tag (bitwise or of TAG_* values)
    ylabel : string
        label of y axis, default is `"lower cell PEs"`
    """
    plt.figure()
    plt.title(title)
    plt.pcolormesh(*np.meshgrid(xedges, yedges), np.log(hist))
    plt.colorbar()
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

def muonScoreLR(xedges, yedges, upper, lower, histLR):
    """
    Calculates the lieklihood ratio muon score.

    Parameters
    ----------
    xedges : array_like
        histogram xedges
    yedges : array_like
        histogram yedges
    upper : array_like
        PEs in upper chamber
    lower : array_like
        PEs in lower chamber
    histLR : array_like
        likelihood ratio of a muon event
    
    Returns
    -------
    ndarray
        muon scores
    """
    uppIdx = np.digitize(upper, xedges)
    lowIdx = np.digitize(lower, yedges)
    uppIdx[uppIdx >= xedges.shape[0]-1] = xedges.shape[0]-2
    lowIdx[lowIdx >= yedges.shape[0]-1] = yedges.shape[0]-2
    muLR = histLR[uppIdx, lowIdx]
    return muLR

def tagShowers(muonTagger, taggedPmtEvts, cut=1, truth=False, ratio=False, makeIds=False, proportion=False):
    """
	Counts muons in showers based on their `muonScore()`.

	Parameters
	----------
    muonTagger : MuonTagger
        the muon tagger to calculate the scores with
    taggedPmtEvts : structured array
        list of PMT events with tags (see `TYPE_TAGGED_PMT_EVTS`)
    cut : float or array_like
        value of muon score above which events are counted as muons;
        can be an array of multiple cuts, default is 1
    truth : bool, optional
        if True, returns true number of muon events, default is False
    ratio : bool, optional
        if True, calculates the electron muon ratio instead of muon counts, default is False
    makeIds : bool, optional
        if True returns shower ids, default is False
    proportion : bool, optional
        if True, calculates the muon proportion instead of muon counts, default is False (ignored if `ratio=True`)

	Returns
	-------    
    cnts : ndarray
        array of muon counts for each shower or multiple arrays if cut is an array
    tCnts : ndarray
        array of true muon counts for each shower (only if truth is True)
    ids : ndarray
        array of shower ids (only if makeIds is True)
	"""
    score = muonTagger.muonScore(taggedPmtEvts)
    # get shower indices
    cdx = taggedPmtEvts["showerID"]
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
        if ratio:
            cumsums = np.cumsum(score <= cuts[i])[idx]
            cnts[i,1:] /= cumsums[1:] - cumsums[0:-1]
            cnts[i,0] /= cumsums[0]
        elif proportion:
            cumsums = np.cumsum(np.ones(cdx.shape))[idx]
            cnts[i,1:] /= cumsums[1:] - cumsums[0:-1]
            cnts[i,0] /= cumsums[0]

    if not isinstance(cut, np.ndarray):
        cnts = cnts[0]
    if not truth: return (cnts, cdx[idx]) if makeIds else cnts
    # sum true muons
    eOnly, muAny = getEMuTags(taggedPmtEvts)
    cumsums = np.cumsum(muAny)[idx]
    tCnts = np.empty(cumsums.shape)
    tCnts[1:] = cumsums[1:] - cumsums[0:-1]
    tCnts[0] = cumsums[0]
    if ratio:
        cumsums = np.cumsum(eOnly)[idx]
        tCnts[1:] /= cumsums[1:] - cumsums[0:-1]
        tCnts[0] /= cumsums[0]
    elif proportion:
        cumsums = np.cumsum(np.ones(cdx.shape))[idx]
        tCnts[1:] /= cumsums[1:] - cumsums[0:-1]
        tCnts[0] /= cumsums[0]
    return (cnts, tCnts, cdx[idx]) if makeIds else (cnts, tCnts)

def magicCumsum(cdx,values,makeIds=False):
    """
    Sums over values with same cdx.
    `cnts = [np.sum(values[cdx == id]) for id in np.unique(cdx)]`

    Parameters
    ----------
    cdx : array_like
        sorted array of repeating event ids
    values : array_like
        values to sum up
    makeIds : bool
        if True returns unique event ids, default is False
    
    Returns
    -------
    cnts : ndarray
        sum of values with same cdx
    ids : ndarray
        unique event ids (only if makeIds is True)
    """
    indices = np.nonzero(np.r_[1, np.diff(cdx)[:-1]])[0]
    idx = np.empty(indices.shape, dtype=int)
    idx[:-1] = indices[1:]-1
    idx[-1] = cdx.size-1
    cumsums = np.cumsum(values)[idx]
    cnts = np.empty(cumsums.shape)
    cnts[1:] = cumsums[1:] - cumsums[0:-1]
    cnts[0] = cumsums[0]
    if makeIds:
        ids = cdx[idx]
        return cnts, ids
    else:
        return cnts

def plotRatioEMu(taggedPmtEvts, primaries, plotProfiles=False):
    """
	Plots the ratio between muons and electrons.
    Figure 1 shows electron only events over muon events.
    Figure 2 shows electron muon ratio over energy.

    Parameters
    ----------
    taggedPmtEvts : structured array
        list of PMT events with tags (see `TYPE_TAGGED_PMT_EVTS`)
    primaries : structured array
        list of primary particles (see `TYPE_PRIMARIES`)
    plotProfiles : bool, optional
        if profile plots should be used (instead of scatter plots)

    Returns
    -------
	eCnt : ndarray
        electron only event counts
    muCnt : ndarray
        muon event counts
	"""
    eOnlyP, muAnyP = getEMuTags(taggedPmtEvts)
    cdx = taggedPmtEvts["showerID"]
    eCnt, ids = magicCumsum(cdx,eOnlyP,True)
    muCnt, ids = magicCumsum(cdx,muAnyP,True)
    selP = primaries["showerType"][ids] == ID_PROTON
    selG = primaries["showerType"][ids] == ID_PHOTON
    #plot into protons/gammas
    plt.figure(1)
    plt.title("Electron muon ratio")
    if plotProfiles:
        plt.errorbar(*profilePoints(muCnt[selP],eCnt[selP]),label="Protons")
        plt.errorbar(*profilePoints(muCnt[selG],eCnt[selG]),label="Gammas")
    else:
        plt.scatter(muCnt[selP],eCnt[selP],label="Protons",marker=".")
        plt.scatter(muCnt[selG],eCnt[selG],label="Gammas",marker="^")
    plt.xlabel("Muon events")
    plt.ylabel("Electron only events")
    plt.legend()
    # plot muon electron ratio
    plt.figure(2)
    plt.title("Electron muon ratio")
    if plotProfiles:
        plt.errorbar(*profilePoints(primaries["showerEnergy"][ids][selP]/1000,muCnt[selP]/eCnt[selP]),label="Protons")
        plt.errorbar(*profilePoints(primaries["showerEnergy"][ids][selG]/1000,muCnt[selG]/eCnt[selG]),label="Gammas")
    else:
        plt.scatter(primaries["showerEnergy"][ids][selP]/1000,muCnt[selP]/eCnt[selP],label="Protons",marker=".")
        plt.scatter(primaries["showerEnergy"][ids][selG]/1000,muCnt[selG]/eCnt[selG],label="Gammas",marker="^")
    plt.xlabel("Energy/TeV")
    plt.ylabel("$N_\mu/N_e$")
    plt.legend()
    return eCnt,muCnt

def plotSeparationCuts(labels, cntsP, cntsG, sep=None, plot=True):
    """
	Calculates and plots cuts to separate gammas and protons.
    Takes two lists of arrays with counts for protons and gammas and plots the
    tagged protons over the tagged gammas for multiple separation cuts.
    Calculates the signal to background ratio for every cut.

    Parameters
    ----------
	labels : array_like
        labels for the created plots
    cntsP : list of array_like
		proton counts (number of arrays matches size of labels)
    cntsG : list of array_like
		gamma counts (number of arrays matches size of labels)
    sep : array_like
        separation cuts to plot, default is `np.linspace(0,200)`
    plot : bool
        if the results should be plotted, default is `True`

    Returns
    -------
	ndarray
		signal to background ratio for every sample (first index) and every
        separation cut (second index), `shape=(labels.size, sep.size)`
	"""
    if not isinstance(sep,np.ndarray):
        sep = np.linspace(0,200)
    sepG = np.empty(sep.shape)
    sepP = np.empty(sep.shape)
    sbr = np.empty((labels.size, sep.size))
    for cp, cg, c, j in zip(cntsP, cntsG, labels, np.arange(labels.size)):
        for i in np.arange(sep.size):
            sepG[i] = (cg > sep[i]).sum()
            sepP[i] = (cp > sep[i]).sum()
        sepG /= cg.shape[0]
        sepP /= cp.shape[0]
        # signal to background ratio
        sbr[j,:] = (1-sepG)/(1-sepP+0.000000001)
        # plot
        if plot:
            plt.plot(sepG, sepP, label="cut="+str(c))
            plt.xlabel("tagged gammas")
            plt.ylabel("tagged protons")
            plt.scatter(sepG,sepP,c=sep)
            plt.legend()
    return sbr


def profilePoints(xs, ys):
    """
	Creates profile plot points with errors from (x,y) data.

	Parameters
	----------
	xs : array_like
		list of x coordinates
	ys : array_like
		list of y coordinates
	
	Returns
	-------
	tuple of ndarray
		x, y, yerr and xerr as used by `plt.errorbar()`
	"""
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

def energyDependentAnalysis(cnts, tCnts, cuts, sep, ids, primaries, plotEdst=True, eBinCnt=4, plotProfiles=True):
    """
	Creates multiple plots for energy dependent shower analysis.
    Figure 4 shows profile lines for muon number-energy dependece.
    Figure 5 shows ROC curves for different energies and cuts.
    Figure 6 shows the optimal cuts for each energy based on SBR.

    Parameters
    ----------
    cnts - ndarray, shape(cuts.size,N)
        array of counts for each cut
    tCnts - ndarray, shape(N)
        array of true counts
    cuts - ndarray
        array of cuts (floats)
    sep - ndarray
        array of separation cuts (counts)
    ids - ndarray, shape(N)
        array of shower ids
    primaries : structured array
        list of primary particles (see `TYPE_PRIMARIES`)
    plotEdst - bool, optional
        plot figure 5, default is True
    eBinCnt - int, optional
        number of energy bins, default is 4
    plotProfiles - bool, optional
        plot figure 4, default is True
	"""
    selP = primaries["showerType"][ids] == ID_PROTON
    selG = primaries["showerType"][ids] == ID_PHOTON
    # plot
    if plotProfiles:
        plt.figure(4)
        plt.errorbar(*profilePoints(primaries["showerEnergy"][ids][selP], tCnts[selP]), label="protons")
        plt.errorbar(*profilePoints(primaries["showerEnergy"][ids][selG], tCnts[selG]), label="gammas")
        plt.title("Muons per shower")
        plt.xlabel("Energy")
        plt.ylabel("Muons")
        plt.legend()

    # energy dependent cuts
    if plotEdst:
        plt.figure(5)
        plt.title("Energy dependent shower tagging")
    energyBins = np.histogram_bin_edges(primaries["showerEnergy"][ids],eBinCnt)
    snrBest = np.empty((energyBins.size-1,5))
    for minE, maxE, i in zip(energyBins[:-1],energyBins[1:], np.arange(snrBest.size)):
        selE = np.logical_and(primaries["showerEnergy"][ids] > minE, primaries["showerEnergy"][ids] < maxE)
        cntsPed = cnts[:,selE&selP]
        cntsGed = cnts[:,selE&selG]
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
    '''TODO fix or remove
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

def plotROC(muAny, muScore, cuts):
    """
	Plots a ROC curve for muons.

	Parameters
	----------
	muAny : array_like
		bool array of true muon events
	muScore : array_like
		score to plot ROC curve for
    cuts : array_like
        cuts to plot
    """
    falseMu = np.zeros(cuts.shape)
    trueMu = np.zeros(cuts.shape)
    for i in np.arange(cuts.shape[0]):
        tagging = muScore > cuts[i]
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

