import numpy as np
from matplotlib import pyplot as plt

TAG_E = 1
TAG_MU = 2
TAG_MESON = 4
TAG_OTHER = 8
def makeTag(parts):
    """
    Calculates particle tags from list of particle IDs.

    Parameters
    ----------
    parts - array_like
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

def makeHistograms(xedges, yedges, taggedPmtEvts):
    """
    Makes histograms from tagged PMT events.

    Parameters
    ----------
    xedges - array_like
        histogram xedges
    yedges - array_like
        histogram yedges
    taggedPmtEvnts - array_like
        list of PMT events with tags (see TODO)
    
    Returns
    -------
    tuple of ndarray
        upper - PEs in upper chamber
        lower - PEs in lower chamber
        muAny - if an muon is in the chamber
        histEOnly - histogram for electron only events
        histMuAny - histogram for muon events
        histLR - likelihood ratio of a muon event
    """
    # extract relevant data
    upper = taggedPmtEvts["upper"]
    lower = taggedPmtEvts["lower"]
    tags = taggedPmtEvts["tagsUpper"]
    tags |= taggedPmtEvts["tagsLower"]
    # select type
    eOnly = tags ^ TAG_E == 0
    #muOnly = tags ^ TAG_MU == 0
    muAny = tags & TAG_MU > 0
    # histogram
    histEOnly, *_ = np.histogram2d(upper[eOnly],lower[eOnly], bins=[xedges,yedges])
    #histMuOnly, xedges, yedges = np.histogram2d(upper[muOnly],lower[muOnly], bins=[xedges,yedges])
    histMuAny, *_ = np.histogram2d(upper[muAny],lower[muAny], bins=[xedges,yedges])
    # scale hists
    histEOnly /= np.sum(histEOnly)
    histMuAny /= np.sum(histMuAny)
    # likelihood ratio
    histLR = histMuAny/histEOnly
    return upper,lower,muAny,histEOnly,histMuAny,histLR

def plotLogHist2d(xedges, yedges, hist, title=None, xlabel="upper cell PEs", ylabel="lower cell PEs"):
    """
    Plots logarithmic 2D histogram.

    Parameters
    ----------
    xedges - array_like
        histogram xedges
    yedges - array_like
        histogram yedges
    hist - array_likeparticle tag (bitwise or of TAG_* values)`
    ylabel - string
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
    xedges - array_like
        histogram xedges
    yedges - array_like
        histogram yedges
    upper - array_like
        PEs in upper chamber
    lower - array_like
        PEs in lower chamber
    histLR - array_like
        likelihood ratio of a muon event
    
    Returns
    -------
    ndarray
        muon scores
    """
    uppIdx = np.digitize(upper, xedges)
    lowIdx = np.digitize(lower, yedges)
    uppIdx[uppIdx >= xedges.shape[0]] = xedges.shape[0]-2
    lowIdx[lowIdx >= yedges.shape[0]] = yedges.shape[0]-2
    muLR = histLR[uppIdx, lowIdx]
    return muLR


def tagShowers(xedges, yedges, taggedPmtEvts, histLR, cut=1, truth=False):
    """
	Counts muons in showers based on their `muonScore()`.

	Parameters
	----------

    xedges - array_like
        histogram xedges
    yedges - array_like
        histogram yedges
    taggedPmtEvnts - array_like
        list of PMT events with tags (see TODO)
    histLR - array_like
        2D likelihood ratio for muon events
    cut - float or array_like
        value of likelihood ratio above which events are counted as muons;
        can be an array of multiple cuts, default is 1
    truth - bool
        if True, returns true number of muon events, default is False

	Returns
	-------
    
    ndarray or tuple of ndarray
        an array of muon counts for each shower or multiple arrays if cut is an array;
        if truth is True the true muon count for each shower is returned as second array
	"""
    upper = taggedPmtEvts["upper"]
    lower = taggedPmtEvts["lower"]
    # calculate muon score
    score = muonScoreLR(xedges, yedges, upper, lower, histLR)
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
    if not isinstance(cut, np.ndarray):
        cnts = cnts[0]
    #TODO handle inf/nan in sums
    if not truth: return cnts
    # sum true muons
    tags = taggedPmtEvts["tagsUpper"].astype(int)
    tags |= taggedPmtEvts["tagsLower"].astype(int)
    muAny = tags & TAG_MU > 0
    cumsums = np.cumsum(muAny)[idx]
    tCnts = np.empty(cumsums.shape)
    tCnts[1:] = cumsums[1:] - cumsums[0:-1]
    tCnts[0] = cumsums[0]
    return cnts, tCnts


def plotSeparationCuts(labels, cntsP, cntsG, sep=None, plot=True):
    """
	Calculates and plots cuts to separate gammas and protons.
    Takes two lists of arrays with counts for protons and gammas and plots the
    tagged protons over the tagged gammas for multiple separation cuts.
    Calculates the signal to background ratio for every cut.

	Parameters
	----------

	labels - array_like
        labels for the created plots
    cntsP - list of array_like
		proton counts (number of arrays matches size of labels)
    cntsG - list of array_like
		gamma counts (number of arrays matches size of labels)
    sep - array_like
        separation cuts to plot, default is `np.linspace(0,200)`
    plot - bool
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
	xs - array_like
		list of x coordinates
	ys - array_like
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