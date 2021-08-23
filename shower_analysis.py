import numpy as np
from matplotlib import pyplot as plt

TYPE_TAGGED_PMT_EVTS = [("showerID","i4"),("pmtIdG","i4"),("upper","i4"),("lower","i4"),("tagsUpper","int8"),("tagsLower","int8"),("distance","f4"),
    ("firstUpper","f4"),("firstLower","f4"),("per10Upper","f4"),("per10Lower","f4"),("per90Upper","f4"),("per90Lower","f4")]
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
    taggedPmtEvnts : array_like
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
    taggedPmtEvnts : array_like
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


def tagShowers(xedges, yedges, taggedPmtEvts, histLR, cut=1, upper="upper", lower="lower", **kwargs):
    """
	Counts muons in showers based on their `muonScore()`.

	Parameters
	----------

    xedges : array_like
        histogram xedges
    yedges : array_like
        histogram yedges
    taggedPmtEvnts : array_like
        list of PMT events with tags (see `TYPE_TAGGED_PMT_EVTS`)
    histLR : array_like
        2D likelihood ratio for muon events
    cut : float or array_like
        value of likelihood ratio above which events are counted as muons;
        can be an array of multiple cuts, default is 1
    truth : bool
        if True, returns true number of muon events, default is False
    upper : string or array_like
        upper events, default is `"upper"`
    lower : string or array_like
        lower events, default is `"lower"`

	Returns
	-------
    
    ndarray or tuple of ndarray
        an array of muon counts for each shower or multiple arrays if cut is an array;
        if truth is True the true muon count for each shower is returned as second array
	"""
    if isinstance(upper,str): upper = taggedPmtEvts[upper]
    if isinstance(lower,str): lower = taggedPmtEvts[lower]
    # calculate muon score
    score = muonScoreLR(xedges, yedges, upper, lower, histLR)
    # tag showers
    return tagShowersS(taggedPmtEvts,score,cut,**kwargs)

def tagShowersMT(muonTagger, taggedPmtEvts, cut=1, **kwargs):
    score = muonTagger.muonScore(taggedPmtEvts)
    return tagShowersS(taggedPmtEvts,score,cut,**kwargs)

def tagShowersS(taggedPmtEvts, score, cut=1, truth=False, ratio=False, makeIds=False):
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
        cnts[i,1:] /= cumsums[1:] - cumsums[0:-1]
        cnts[i,0] /= cumsums[0]
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

