import uproot
import awkward as ak
import numpy as np
from shower_analysis import TAG_E, TAG_MU, TAG_MESON, TAG_OTHER

# detector geometry
pmtIdG, east, north = np.loadtxt("data/swgo_reference_survey.txt", unpack=True)

taggedPmtEvts = []
tpedt = [("showerID","i4"),("pmtIdG","i4"),("upper","i4"),("lower","i4"),("tagsUpper","int8"),("tagsLower","int8"),("distance","f4")]
path = "data/gammabbww/"
for batch,report in uproot.iterate([path+"*.root:XCDF"],report=True,
        filter_name=["HAWCSim.Evt.Num","HAWCSim.PE.Time","HAWCSim.Evt.firstTime","HAWCSim.PE.PMTID","HAWCSim.Evt.X","HAWCSim.Evt.Y","HAWCSim.PE.parPType"]):
    peTimes = batch["HAWCSim.PE.Time"]-batch["HAWCSim.Evt.firstTime"]
    for i in np.arange(ak.num(peTimes,0)):
        pmtIds = batch["HAWCSim.PE.PMTID"][i].to_numpy()
        partIds = batch["HAWCSim.PE.parPType"][i].to_numpy()
        # count upper/lower
        pmtIdsU, pmtCnts = np.unique(pmtIds, return_counts=True)
        upper = np.zeros(pmtIdG.shape)
        lower = np.zeros(pmtIdG.shape)
        sel = (pmtIdsU[pmtIdsU%2==0]/2).astype(int)
        upper[sel] = pmtCnts[pmtIdsU%2==0]
        sel = (pmtIdsU[pmtIdsU%2==1]/2).astype(int)
        lower[sel] = pmtCnts[pmtIdsU%2==1]
        # tag events
        tagsUpper = np.zeros(pmtIdG.shape,dtype=np.int8)
        tagsLower = np.zeros(pmtIdG.shape,dtype=np.int8)
        pis = partIds[pmtIds%2==0]
        # tagsUpper[(pmtIds[pmtIds%2==0]/2).astype(int)] |= (pis < 4)*TAG_E | (np.logical_and(5 <= pis, pis < 7))*TAG_MU | (np.logical_and(7 <= pis, pis < 13))*TAG_MESON | (pis >= 13)*TAG_OTHER
        np.bitwise_or.at(tagsUpper, (pmtIds[pmtIds%2==0]/2).astype(int),
            (pis < 4)*TAG_E | (np.logical_and(5 <= pis, pis < 7))*TAG_MU | (np.logical_and(7 <= pis, pis < 13))*TAG_MESON | (pis >= 13)*TAG_OTHER)
        pis = partIds[pmtIds%2==1]
        #tagsLower[(pmtIds[pmtIds%2==1]/2).astype(int)] |= (pis < 4)*TAG_E | (np.logical_and(5 <= pis, pis < 7))*TAG_MU | (np.logical_and(7 <= pis, pis < 13))*TAG_MESON | (pis >= 13)*TAG_OTHER
        np.bitwise_or.at(tagsLower, (pmtIds[pmtIds%2==1]/2).astype(int),
            (pis < 4)*TAG_E | (np.logical_and(5 <= pis, pis < 7))*TAG_MU | (np.logical_and(7 <= pis, pis < 13))*TAG_MESON | (pis >= 13)*TAG_OTHER)
        u = np.unique(tagsLower)
        # calculate distance
        dist = np.hypot(east-batch["HAWCSim.Evt.X"][i], north-batch["HAWCSim.Evt.Y"][i])
        # drop empty
        sel = np.logical_or(upper > 0,lower > 0)
        app = np.empty(sel.sum(), dtype=tpedt)
        app["showerID"] = (i+report.start)*np.ones(pmtIdG[sel].shape)
        app["pmtIdG"] = pmtIdG[sel]
        app["upper"] = upper[sel]
        app["lower"] = lower[sel]
        app["tagsUpper"] = tagsUpper[sel]
        app["tagsLower"] = tagsLower[sel]
        app["distance"] = dist[sel]
        taggedPmtEvts.append(app)

    break

# save tagged events
saveTaggedEvts = True
if saveTaggedEvts:
    data = np.concatenate(taggedPmtEvts,axis=-1)
    np.save(path+"taggedPmtEvts2.npy",data)
