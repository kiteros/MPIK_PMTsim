import uproot
import numpy as np
from matplotlib import pyplot as plt

# detector geometry
pmtIdG, east, north = np.loadtxt("data/swgo_reference_survey.txt", unpack=True)
makePlots = False
if makePlots:
    plt.figure(1)
    plt.scatter(east, north, c="grey")

# shower events
events = uproot.lazy("data/*.root:XCDF")
binsAll = np.linspace(-5,170)
histAll = np.zeros(binsAll.shape[0]-1)
upperAll = np.zeros(pmtIdG.shape)
lowerAll = np.zeros(pmtIdG.shape)
particles = np.zeros(70)
parTicks = ([2,3,5,6,8,9,11,12,14,15],["$e^+$","$e^-$","$\mu^+$","$\mu^-$","$\pi^+$","$\pi^-$","$K^+$","$K^-$","$p$","$\\bar p$"])
cdx = 0
for pmtIds, peTimes, ft, ex, ey, parPType in zip(events["HAWCSim.PE.PMTID"], events["HAWCSim.PE.Time"], events["HAWCSim.Evt.firstTime"], events["HAWCSim.Evt.X"], events["HAWCSim.Evt.Y"], events["HAWCSim.PE.parPType"]):
    # plot all
    peTimes = peTimes.to_numpy()-ft
    pmtIds = pmtIds.to_numpy()
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
    if makePlots:
        plt.figure(13)
        plt.scatter(upper,lower)
        #plt.xscale("log")
        #plt.yscale("log")
    # get particle types
    parPType = parPType.to_numpy()
    pTypes, pCnts = np.unique(parPType, return_counts=True)
    particles[pTypes] += pCnts
    if makePlots:
        plt.figure(3)
        plt.bar(pTypes, pCnts)
        plt.xticks(*parTicks)
    # carry over
    histAll += hist
    upperAll += upper
    lowerAll += lower
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
hist, xedges, yedges = np.histogram2d(upperAll,lowerAll, 100)
plt.pcolormesh(*np.meshgrid(xedges, yedges), np.log(hist+1))
plt.colorbar()
plt.xscale("log")
plt.yscale("log")
#np.save("data/is_upper.npy",upperAll)
#np.save("data/is_lower.npy",lowerAll)
# particles
plt.figure(3)
plt.bar(np.arange(16), particles[:16]) # ignore resonaces and neutrinos
plt.xticks(*parTicks)

plt.show()
