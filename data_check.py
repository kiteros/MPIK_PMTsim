import uproot
import numpy as np
from matplotlib import pyplot as plt

# detector geometry
pmtIdG, east, north = np.loadtxt("data/swgo_reference_survey.txt", unpack=True)
plt.figure(1)
plt.scatter(east, north, c="grey")

# shower events
events = uproot.lazy("data/DAT000001.root:XCDF")
for pmtIds, peTimes, ft, ex, ey in zip(events["HAWCSim.PE.PMTID"], events["HAWCSim.PE.Time"], events["HAWCSim.Evt.firstTime"], events["HAWCSim.Evt.X"], events["HAWCSim.Evt.Y"]):
    # plot all
    peTimes = peTimes.to_numpy()-ft
    pmtIds = pmtIds.to_numpy()
    plt.figure(2)
    plt.scatter(peTimes, pmtIds, c="grey")
    # plot with geometry
    pmtIdP, cntP = np.unique(pmtIds, return_counts=True)
    cntP = np.log(cntP)
    cm = plt.cm.get_cmap("YlOrRd")
    plt.figure(1)
    plt.scatter(east[(pmtIdP/2).astype(int)], north[(pmtIdP/2).astype(int)], vmin=cntP.min(), vmax=cntP.max(), c=cntP, cmap=cm)
    plt.scatter(ex,ey,marker="x")
    plt.gca().add_patch(plt.Circle((ex,ey),20*100,fill=False))
    # make time histogram
    hist, bins = np.histogram(peTimes, np.linspace(-5,170))
    plt.figure(7)
    plt.plot(bins[:-1], hist)
    plt.yscale("log")
    # get events from upper/lower cell outside exclusion area
    two = pmtIdP[pmtIdP%2==0]
    one = pmtIdP[pmtIdP%2==1]
    print(one,two)
    break


plt.show()
