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
events = uproot.lazy("data/DAT000001.root:XCDF")
binsAll = np.linspace(-5,170)
histAll = np.zeros(binsAll.shape[0]-1)
upperAll = np.zeros(pmtIdG.shape)
lowerAll = np.zeros(pmtIdG.shape)
cdx = 0
for pmtIds, peTimes, ft, ex, ey in zip(events["HAWCSim.PE.PMTID"], events["HAWCSim.PE.Time"], events["HAWCSim.Evt.firstTime"], events["HAWCSim.Evt.X"], events["HAWCSim.Evt.Y"]):
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
    # carry over
    histAll += hist
    upperAll += upper
    lowerAll += lower
    cdx += 1
    #if cdx > 5: break

print(cdx)
plt.figure(7)
plt.plot(binsAll[:-1], histAll)
plt.yscale("log")
plt.figure(13)
hist, xedges, yedges = np.histogram2d(upperAll,lowerAll, 100)
plt.pcolormesh(*np.meshgrid(xedges, yedges), np.log(hist))
plt.colorbar()
np.save("data/is_upper.npy",upperAll)
np.save("data/is_lower.npy",lowerAll)

plt.show()
