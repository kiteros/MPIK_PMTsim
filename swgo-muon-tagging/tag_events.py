from muon_tagging import loadTagger
from shower_analysis import *
from sklearn.metrics import roc_curve
import argparse
import numpy as np
from sys import float_info
from matplotlib import pyplot as plt

# --- start ---
if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser(description="Creates simple muon taggers.")
    parser.add_argument("-n", "--name", help="name of tagger (including directory name)", default="models/mu_tag_LR_charge")
    parser.add_argument("-e", "--exclusion", type=float, default=0, help="exclusion region in m (default is 0)")
    parser.add_argument("-a", "--analysis", action="store_true", help="perform energy dependent analysis (EDA)")
    parser.add_argument("-r", "--roc", action="store_true", help="plot ROC curve")
    parser.add_argument("--profiles", action="store_true", help="plot muon per shower profile plot (EDA)")
    parser.add_argument("--percentiles", action="store_true", help="plot muon per shower percentile plot (EDA)")
    parser.add_argument("--sroc", action="store_true", help="plot ROC curves for showers, only use with -b and -c (EDA)")
    parser.add_argument("-c", "--cuts", type=float, default=None, help="muon score cuts for EDA", nargs="+", metavar="C")
    parser.add_argument("-b", "--bins", type=int, default=20, help="number of energy bins for EDA")
    parser.add_argument("paths", help="list of directories containing tagged event data", nargs="+")
    args = parser.parse_args()
    # load tagger
    mt = loadTagger(args.name)
    # load data
    taggedPmtEvts, primaries = loadData(args.paths,args.exclusion)

    # plot roc curve
    if args.roc:
        _, muAny = getEMuTags(taggedPmtEvts)
        plt.figure()
        plt.title("ROC curve")
        plt.xlabel("false muons")
        plt.ylabel("true muons")
        y_pred = mt.muonScore(taggedPmtEvts)
        y_pred = np.nan_to_num(y_pred, nan=float_info.min, neginf=float_info.min, posinf=float_info.max)
        fpr, tpr, cuts = roc_curve(muAny, y_pred)
        plt.plot(fpr,tpr)

    # energy dependent analysis
    if args.analysis:
        cuts = np.array(args.cuts) if args.cuts else np.linspace(0,40,50)
        sep = np.logspace(-10,0,200)
        cnts, tCnts, ids = tagShowers(mt, taggedPmtEvts, cut=cuts, truth=True, ratio=False, makeIds=True, proportion=True)
        selP = primaries["showerType"][ids] == ID_PROTON
        selG = primaries["showerType"][ids] == ID_PHOTON
        energyDependentAnalysis(cnts, tCnts, cuts, sep, ids, primaries, args.sroc, args.bins, args.profiles, args.percentiles)


    plt.show()


