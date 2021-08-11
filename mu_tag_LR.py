import numpy as np
from matplotlib import pyplot as plt
from quick_hist import makeHistograms, muonScoreLR

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

# calculate muon score
score = muonScoreLR(xedges,yedges, upper, lower, histLR)

# tag showers
cdx = taggedPmtEvts[0]
showers = np.arange(cdx.min(), cdx.max())
sums = np.zeros(showers.shape)
cnts = np.zeros(showers.shape)
for i in np.arange(showers.shape[0]):
    ss = score[cdx == showers[i]]
    sums[i] = np.sum(ss)
    cnts[i] = np.sum(ss > 1)

# plot
plt.figure(1)
plt.hist(sums, label="protons")
plt.figure(2)
plt.hist(cnts, label="protons")

plt.show()
