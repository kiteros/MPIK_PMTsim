import uproot
import numpy as np
from matplotlib import pyplot as plt

TAG_E = 1
TAG_MU = 2
TAG_MESON = 4
TAG_OTHER = 8
def makeTag(parts):
    tag = 0
    for p in parts:
        if p < 4: tag |= TAG_E
        elif p >= 5 and p < 7: tag |= TAG_MU
        elif p >= 7 and p < 13: tag |= TAG_MESON
        else: tag |= TAG_OTHER
    return tag

# detector geometry
pmtIdG, east, north = np.loadtxt("data/swgo_reference_survey.txt", unpack=True)
makePlots = False
if makePlots:
    plt.figure(1)
    plt.scatter(east, north, c="grey")

# shower events
path = "data/protonbbww/" # gamma 5, 46 broken?
events = uproot.lazy(path+"*.root:XCDF")
binsAll = np.linspace(-5,170)
histAll = np.zeros(binsAll.shape[0]-1)
xedges = np.linspace(10**0,10**4,1001)
yedges = np.linspace(10**0,10**4,1001)
histULAll = np.zeros((xedges.shape[0]-1,yedges.shape[0]-1))
taggedPmtEvts = []
particles = np.zeros(70)
parTicks = ([2,3,5,6,8,9,11,12,14,15],["$e^+$","$e^-$","$\mu^+$","$\mu^-$","$\pi^+$","$\pi^-$","$K^+$","$K^-$","$p$","$\\bar p$"])
cdx = 0
for pmtIds, peTimes, ft, ex, ey, parPType in zip(events["HAWCSim.PE.PMTID"], events["HAWCSim.PE.Time"], events["HAWCSim.Evt.firstTime"], events["HAWCSim.Evt.X"], events["HAWCSim.Evt.Y"], events["HAWCSim.PE.parPType"]):
    # plot all
    peTimes = peTimes.to_numpy()-ft
    pmtIds = pmtIds.to_numpy()
    parPType = parPType.to_numpy()
    if makePlots:
        plt.figure(2)
        plt.scatter(peTimes, pmtIds, c="grey")
    # plot with geometry
    pmtIdP, cntP = np.unique(pmtIds, return_counts=True)
    cntPL = np.log(cntP)
    if makePlots:
        cm = plt.cm.get_cmap("YlOrRd")
        plt.figure(1)
        plt.scatter(east[(pmtIdP/2).astype(int)], north[(pmtIdP/2).astype(int)], vmin=cntPL.min(), vmax=cntPL.max(), c=cntPL, cmap=cm)
        plt.scatter(ex,ey,marker="x")
        plt.gca().add_patch(plt.Circle((ex,ey),20*100,fill=False))
    # tag pmt signals
    tagsUpper = np.zeros(pmtIdG.shape,dtype=int)
    tagsLower = np.zeros(pmtIdG.shape,dtype=int)
    for id in pmtIdP:
        if id%2 == 0:
            tagsUpper[int(id/2)] |= makeTag(np.unique(parPType[pmtIds == id]))
        if id%2 == 1:
            tagsLower[int(id/2)] |= makeTag(np.unique(parPType[pmtIds == id]))
    # make time histogram
    hist, bins = np.histogram(peTimes, binsAll)
    if makePlots:
        plt.figure(7)
        plt.plot(bins[:-1], hist)
        plt.yscale("log")
    # get events from upper/lower cell outside exclusion area
    upper = np.zeros(pmtIdG.shape)
    lower = np.zeros(pmtIdG.shape)
    exclusion = east**2 + north**2 < (20*100)**2
    sel = (pmtIdP[pmtIdP%2==0]/2).astype(int)
    upper[sel] = cntP[pmtIdP%2==0]
    sel = (pmtIdP[pmtIdP%2==1]/2).astype(int)
    lower[sel] = cntP[pmtIdP%2==1]
    upper[exclusion] = 0
    lower[exclusion] = 0
    histUL, xedges, yedges = np.histogram2d(upper,lower, bins=[xedges,yedges])
    if makePlots:
        plt.figure(13)
        plt.scatter(upper,lower)
        #plt.xscale("log")
        #plt.yscale("log")
    # get particle types
    pTypes, pCnts = np.unique(parPType, return_counts=True)
    particles[pTypes] += pCnts
    if makePlots:
        plt.figure(3)
        plt.bar(pTypes, pCnts)
        plt.xticks(*parTicks)
    # carry over
    histAll += hist
    histULAll += histUL
    sel = np.logical_or(upper > 0,lower > 0)
    taggedPmtEvts.append((cdx*np.ones(pmtIdG[sel].shape), pmtIdG[sel], upper[sel], lower[sel], tagsUpper[sel], tagsLower[sel]))
    cdx += 1
    #if cdx > 5: break
    #break

print(cdx)
# fig 7 (times)
plt.figure(7)
plt.plot(binsAll[:-1], histAll)
plt.yscale("log")
# fig 13 (upper/lower ratio)
plt.figure(13)
plt.pcolormesh(*np.meshgrid(xedges, yedges), np.log(histULAll))
plt.colorbar()
plt.xscale("log")
plt.yscale("log")
#np.save(path+"histUL.npy",histULAll)
#np.save(path+"xedges.npy",xedges)
#np.save(path+"yedges.npy",yedges)
# particles
plt.figure(3)
plt.bar(np.arange(16), particles[:16]) # ignore resonaces and neutrinos
plt.xticks(*parTicks)

# save tagged events
saveTaggedEvts = False
if saveTaggedEvts:
    cols = 6
    rows = 0
    for e in taggedPmtEvts:
        rows += e[0].size
    data = np.zeros((cols, rows))
    off = 0
    for e in taggedPmtEvts:
        newOff = off + e[0].shape[0]
        for i in np.arange(cols):
            data[i,off:newOff] = e[i]
        off = newOff
    np.save(path+"taggedPmtEvts.npy",data)



plt.show()
