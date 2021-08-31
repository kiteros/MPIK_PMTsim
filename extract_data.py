import uproot
import awkward as ak
import numpy as np
from shower_analysis import TAG_E, TAG_MU, TAG_MESON, TAG_OTHER, TYPE_PRIMARIES, TYPE_TAGGED_PMT_EVTS
import argparse

# parse command line arguments
parser = argparse.ArgumentParser(description="Extracts PMT event data from shower simulations.")
parser.add_argument("-e", "--events", action="store_true", help="create PMT event list")
parser.add_argument("-p", "--primaries", action="store_true", help="create primary particle list")
parser.add_argument("-g", "--geometry", default="data/swgo_reference_survey.txt", help="name of geometry file")
parser.add_argument("path", default="/", help="path of the .root files", nargs="?")
args = parser.parse_args()

# detector geometry
pmtIdG, east, north = np.loadtxt(args.geometry, unpack=True)

# batch process .root files
taggedPmtEvts = []
primaries = []
path = args.path
for batch,report in uproot.iterate([path+"*.root:XCDF"],report=True,
        filter_name=["HAWCSim.Evt.Num","HAWCSim.PE.Time","HAWCSim.Evt.firstTime","HAWCSim.PE.PMTID","HAWCSim.Evt.X","HAWCSim.Evt.Y","HAWCSim.PE.parPType"
        ,"HAWCSim.Evt.pType","HAWCSim.Evt.Energy"]):
    peTimes = batch["HAWCSim.PE.Time"]-batch["HAWCSim.Evt.firstTime"]
    for i in np.arange(ak.num(peTimes,0)):
        # make numpy arrays
        pmtIds = batch["HAWCSim.PE.PMTID"][i].to_numpy()
        partIds = batch["HAWCSim.PE.parPType"][i].to_numpy()
        pets = peTimes[i].to_numpy()
        # shower info
        primaries.append(np.array([(i+report.start,batch["HAWCSim.Evt.pType"][i],batch["HAWCSim.Evt.Energy"][i])],dtype=TYPE_PRIMARIES))
        # skip if empty (this actually happens!)
        if pets.size == 0 or ~args.events: continue
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
        percentiles = np.array([0.1,0.2,0.5,0.9])
        perX = [pets[(n*p+indices).astype(int)] for p in percentiles]
        # bring to pmtIdG shape
        uspids = spids[indices]
        #TODO good idea?
        upperFirst = np.zeros(pmtIdG.shape)
        upperFirst[(uspids[uspids%2==0]/2).astype(int)] = first[uspids%2==0]
        lowerFirst = np.zeros(pmtIdG.shape)
        lowerFirst[(uspids[uspids%2==1]/2).astype(int)] = first[uspids%2==1]
        upperPerX = np.zeros((percentiles.shape[0],pmtIdG.shape[0]))
        lowerPerX = np.zeros((percentiles.shape[0],pmtIdG.shape[0]))
        for j in np.arange(percentiles.size):
            upperPerX[j,(uspids[uspids%2==0]/2).astype(int)] = perX[j][uspids%2==0]
            lowerPerX[j,(uspids[uspids%2==1]/2).astype(int)] = perX[j][uspids%2==1]
        # pass non empty
        sel = np.logical_or(upper > 0,lower > 0)
        app = np.empty(sel.sum(), dtype=TYPE_TAGGED_PMT_EVTS)
        app["showerID"] = (i+report.start)*np.ones(pmtIdG[sel].shape)
        app["pmtIdG"] = pmtIdG[sel]
        app["upper"] = upper[sel]
        app["lower"] = lower[sel]
        app["tagsUpper"] = tagsUpper[sel]
        app["tagsLower"] = tagsLower[sel]
        app["distance"] = dist[sel]
        app["firstUpper"] = upperFirst[sel]
        app["firstLower"] = lowerFirst[sel]
        for p, j in zip(percentiles,np.arange(percentiles.size)):
            app["per{:0.0f}Upper".format(p*100)] = upperPerX[j][sel]
            app["per{:0.0f}Lower".format(p*100)] = lowerPerX[j][sel]
        taggedPmtEvts.append(app)
    #break

# save tagged events
if args.events:
    data = np.concatenate(taggedPmtEvts,axis=-1)
    np.save(path+"taggedPmtEvts2.npy",data)
# save primaries
if args.primaries:
    data = np.concatenate(primaries,axis=-1)
    np.save(path+"primaries.npy",data)
