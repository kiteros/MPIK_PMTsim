import argparse
from muon_tagging import MuTagLR, MuTagRise
from shower_analysis import loadData, makeHistograms
import numpy as np

def makeRiseTimeTagger(taggedPmtEvents, xedges=None, yedges=None, high="50", low="10", name=None):
    """
    Minimal setup of an MuTagRise muon tagger.

    Parameters
    ----------
    taggedPmtEvts : structured array
        list of PMT events with tags (see `TYPE_TAGGED_PMT_EVTS`)
    xedges : ndarray, optional
        histogram xedges, if None uses `np.geomspace(0.01,500,101)`
    yedges : ndarray, optional
        histogram yedges, if None uses `np.geomspace(0.01,500,101)`
    high : string, optional
        higher percentile to use (two digit string), default is 50
    low : string, optional
        lower percentile to use (two digit string), default is 10
    name : string, optional
        name of the tagger (may include directory name)
        if no name is given, a name is automatically generated and tagger is saved in `models/`
    """
    diffUpper = taggedPmtEvents["per"+high+"Upper"]-taggedPmtEvents["per"+low+"Upper"]
    diffLower = taggedPmtEvents["per"+high+"Lower"]-taggedPmtEvents["per"+low+"Lower"]
    if xedges==None: xedges = np.geomspace(0.01,500,101)
    if yedges==None: yedges = np.geomspace(0.01,500,101)
    *_, histLR = makeHistograms(xedges,yedges,taggedPmtEvents,diffUpper,diffLower)
    mt = MuTagRise(xedges,yedges,histLR,low=low,high=high)
    if name==None:
        name = "models/mu_tag_LR_rise_"+high+"_"+low
    mt.save(name)

def makeChargeTagger(taggedPmtEvents, xedges=None, yedges=None, name=None):
    """
    Minimal setup of an MuTagLR muon tagger.

    Parameters
    ----------
    taggedPmtEvts : structured array
        list of PMT events with tags (see `TYPE_TAGGED_PMT_EVTS`)
    xedges : ndarray, optional
        histogram xedges, if None uses `np.linspace(10**0,10**4,1001)`
    yedges : ndarray, optional
        histogram yedges, if None uses `np.linspace(10**0,10**4,1001)`
    name : string, optional
        name of the tagger (may include directory name)
        if no name is given, a name is automatically generated and tagger is saved in `models/`
    """
    if xedges==None: xedges = np.linspace(10**0,10**4,1001)
    if yedges==None: yedges = np.linspace(10**0,10**4,1001)
    *_, histLR = makeHistograms(xedges,yedges,taggedPmtEvents)
    mt = MuTagLR(xedges,yedges,histLR)
    if name==None:
        name = "models/mu_tag_LR_charge"
    mt.save(name)

# --- start ---
if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser(description="Creates simple muon taggers.")
    parser.add_argument("-c", "--charge", action="store_true", help="create charge tagger")
    parser.add_argument("-r", "--rise", action="store_true", help="create rise time tagger")
    parser.add_argument("-e", "--exclusion", type=float, default=0, help="exclusion region in m (default is 0)")
    parser.add_argument("-n", "--name", help="name of tagger (including directory name)")
    parser.add_argument("paths", help="list of directories containing relevant data", nargs="+")
    args = parser.parse_args()
    if args.charge and args.rise and args.name:
        raise ValueError("Cannot specify same name for multiple taggers!")
    # load data
    tpe, _ = loadData(args.paths,args.exclusion)
    # make tagger
    if args.charge:
        makeChargeTagger(tpe,name=args.name)
    if args.rise:
        makeRiseTimeTagger(tpe,name=args.name)
