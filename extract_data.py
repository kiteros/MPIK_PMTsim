import uproot
import awkward as ak
import numpy as np
from shower_analysis import TAG_E, TAG_MU, TAG_MESON, TAG_OTHER

# detector geometry
pmtIdG, east, north = np.loadtxt("data/swgo_reference_survey.txt", unpack=True)

taggedPmtEvts = []
tpedt = [("showerID","i4"),("pmtIdG","i4"),("upper","i4"),("lower","i4"),("tagsUpper","int8"),("tagsLower","int8"),("distance","f4"),
    ("firstUpper","f4"),("firstLower","f4"),("per10Upper","f4"),("per10Lower","f4"),("per90Upper","f4"),("per90Lower","f4")]
primaries = []
pridt = [("showerID","i4"),("showerType","i4"),("showerEnergy","f4")]
path = "data/protonbbww/"
for batch,report in uproot.iterate([path+"*.root:XCDF"],report=True,
        filter_name=["HAWCSim.Evt.Num","HAWCSim.PE.Time","HAWCSim.Evt.firstTime","HAWCSim.PE.PMTID","HAWCSim.Evt.X","HAWCSim.Evt.Y","HAWCSim.PE.parPType"
        ,"HAWCSim.Evt.pType","HAWCSim.Evt.Energy"]):
    peTimes = batch["HAWCSim.PE.Time"]-batch["HAWCSim.Evt.firstTime"]
    for i in np.arange(ak.num(peTimes,0)):
        # make numpy arrays
        pmtIds = batch["HAWCSim.PE.PMTID"][i].to_numpy()
        partIds = batch["HAWCSim.PE.parPType"][i].to_numpy()
        pets = peTimes[i].to_numpy()
        # skip if empty
        if pets.size == 0: continue
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
        #tagsUpper[(pmtIds[pmtIds%2==0]/2).astype(int)] |= (pis < 4)*TAG_E | (np.logical_and(5 <= pis, pis < 7))*TAG_MU | (np.logical_and(7 <= pis, pis < 13))*TAG_MESON | (pis >= 13)*TAG_OTHER
        np.bitwise_or.at(tagsUpper, (pmtIds[pmtIds%2==0]/2).astype(int),
            (pis < 4)*TAG_E | (np.logical_and(5 <= pis, pis < 7))*TAG_MU | (np.logical_and(7 <= pis, pis < 13))*TAG_MESON | (pis >= 13)*TAG_OTHER)
        pis = partIds[pmtIds%2==1]
        #tagsLower[(pmtIds[pmtIds%2==1]/2).astype(int)] |= (pis < 4)*TAG_E | (np.logical_and(5 <= pis, pis < 7))*TAG_MU | (np.logical_and(7 <= pis, pis < 13))*TAG_MESON | (pis >= 13)*TAG_OTHER
        np.bitwise_or.at(tagsLower, (pmtIds[pmtIds%2==1]/2).astype(int),
            (pis < 4)*TAG_E | (np.logical_and(5 <= pis, pis < 7))*TAG_MU | (np.logical_and(7 <= pis, pis < 13))*TAG_MESON | (pis >= 13)*TAG_OTHER)
        u = np.unique(tagsLower)
        # calculate distance
        dist = np.hypot(east-batch["HAWCSim.Evt.X"][i], north-batch["HAWCSim.Evt.Y"][i])
        # calculate percentiles
        sidx = np.lexsort((pets,pmtIds))
        pets = pets[sidx]
        spids = pmtIds[sidx] #TODO use instead of pmtIds?
        indices = np.nonzero(np.r_[1, np.diff(spids)[:-1]])[0]
        n = np.empty(indices.size)
        n[:-1] = indices[1:]-indices[:-1]
        n[-1] = spids.size-indices[-1] #TODO is this the same as pmtCnts?
        first = pets[indices]
        per10 = pets[(n*0.1+indices).astype(int)]
        per90 = pets[(n*0.9+indices).astype(int)]
        # bring to pmtIdG shape
        uspids = spids[indices]
        #TODO good idea?
        upperFirst = np.zeros(pmtIdG.shape)
        upperFirst[(uspids[uspids%2==0]/2).astype(int)] = first[uspids%2==0]
        lowerFirst = np.zeros(pmtIdG.shape)
        lowerFirst[(uspids[uspids%2==1]/2).astype(int)] = first[uspids%2==1]
        upperPer10 = np.zeros(pmtIdG.shape)
        upperPer10[(uspids[uspids%2==0]/2).astype(int)] = per10[uspids%2==0]
        lowerPer10 = np.zeros(pmtIdG.shape)
        lowerPer10[(uspids[uspids%2==1]/2).astype(int)] = per10[uspids%2==1]
        upperPer90 = np.zeros(pmtIdG.shape)
        upperPer90[(uspids[uspids%2==0]/2).astype(int)] = per90[uspids%2==0]
        lowerPer90 = np.zeros(pmtIdG.shape)
        lowerPer90[(uspids[uspids%2==1]/2).astype(int)] = per90[uspids%2==1]
        # pass non empty
        sel = np.logical_or(upper > 0,lower > 0)
        app = np.empty(sel.sum(), dtype=tpedt)
        app["showerID"] = (i+report.start)*np.ones(pmtIdG[sel].shape)
        app["pmtIdG"] = pmtIdG[sel]
        app["upper"] = upper[sel]
        app["lower"] = lower[sel]
        app["tagsUpper"] = tagsUpper[sel]
        app["tagsLower"] = tagsLower[sel]
        app["distance"] = dist[sel]
        app["firstUpper"] = upperFirst[sel]
        app["firstLower"] = lowerFirst[sel]
        app["per10Upper"] = upperPer10[sel]
        app["per10Lower"] = lowerPer10[sel]
        app["per90Upper"] = upperPer90[sel]
        app["per90Lower"] = lowerPer90[sel]
        taggedPmtEvts.append(app)
        # shower info
        primaries.append((i+report.start,batch["HAWCSim.Evt.pType"][i],batch["HAWCSim.Evt.Energy"][i]))
    #break

# save tagged events
saveTaggedEvts = False
if saveTaggedEvts:
    data = np.concatenate(taggedPmtEvts,axis=-1)
    np.save(path+"taggedPmtEvts2.npy",data)
# save primaries
savePrimaries = False
if saveTaggedEvts:
    data = np.concatenate(primaries,axis=-1)
    np.save(path+"primaries.npy",data)
