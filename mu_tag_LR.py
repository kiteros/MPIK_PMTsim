import numpy as np
from matplotlib import pyplot as plt
from quick_hist import makeHistograms, muonScoreLR, TAG_MU, plotLogHist2d

# inputs
protons = "data/protonbbww/"
gammas = "data/gammabbww/"

# load edges
xedges = np.load(protons+"xedges.npy")
yedges = np.load(protons+"yedges.npy")

# load events
taggedPmtEvts = np.load(protons+"taggedPmtEvts.npy")

# calculate likelihood ratio
xedges, yedges, upper, lower, *_, histLR = makeHistograms(xedges, yedges, taggedPmtEvts)

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

# tag events
cuts = np.linspace(5,20,4)
cntsP, tCntsP = tagShowers(xedges, yedges, taggedPmtEvts, histLR, cut=cuts, truth=True)

# repeat for gammas
taggedPmtEvts = np.load(gammas+"taggedPmtEvts.npy")
cntsG, tCntsG = tagShowers(xedges, yedges, taggedPmtEvts, histLR, cut=cuts, truth=True)

# plot
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
plt.legend()

# find separation
plt.figure(3)
plt.title("Shower tagging")
sep = np.linspace(0,10)
sepG = np.empty(sep.shape)
sepP = np.empty(sep.shape)
for cp, cg, c in zip(cntsP, cntsG, cuts):
    for i in np.arange(sep.size):
        sepG[i] = (cg > sep[i]).sum()
        sepP[i] = (cp > sep[i]).sum()
    sepG /= cg.shape[0]
    sepP /= cp.shape[0]
    # plot
    plt.plot(sepG, sepP, label="cut="+str(c))
    plt.xlabel("tagged gammas")
    plt.ylabel("tagged protons")
    plt.scatter(sepG,sepP,c=sep)
    plt.legend()
plt.colorbar(label="minimum muons per shower")

# energy distribution
energyP = np.load(protons+"energy.npy")
energyG = np.load(gammas+"energy.npy")
#TODO get proton energies
#plt.plot(energyP,tCntsP,label="protons")
#hist, xe, ye = np.histogram2d(energyG,tCntsG)
#plotLogHist2d(xe, ye, hist, xlabel="Energy", ylabel="Muons")
bins = np.histogram_bin_edges(energyG)
x = np.empty(bins.size-1)
xerr = np.empty(x.shape)
y = np.empty(x.shape)
yerr = np.empty(x.shape)
for i in np.arange(x.size):
    sel = np.logical_and(energyG > bins[i],energyG <= bins[i+1])
    x[i] = np.mean(energyG[sel])
    xerr[i] = np.std(energyG[sel])
    y[i] = np.mean(tCntsG[sel])
    yerr[i] = np.std(tCntsG[sel])
plt.figure(4)
plt.errorbar(x,y,yerr,xerr)
plt.xlabel("Energy")
plt.xlabel("Muons")


plt.show()
