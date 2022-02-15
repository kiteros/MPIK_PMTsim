from sys import float_info
from muon_tagging import *
import numpy as np
from matplotlib import pyplot as plt
from shower_analysis import *
from matplotlib.patches import Rectangle
from sklearn.metrics import roc_curve


def plotTimes(mtRT, p, taggedPmtEvts):
    diffUpper = taggedPmtEvts["per" + p + "Upper"] - taggedPmtEvts["per10Upper"]
    diffLower = taggedPmtEvts["per" + p + "Lower"] - taggedPmtEvts["per10Lower"]
    *_, histEOnly, histMuAny, histLR = makeHistograms(mtRT.xedges, mtRT.yedges, taggedPmtEvts, diffUpper, diffLower)
    plotLogHist2d(
        mtRT.xedges,
        mtRT.yedges,
        histEOnly,
        "Electron only (box)",
        "upper time 10-" + p + "/ns",
        "lower time 10-" + p + "/ns",
    )
    plotLogHist2d(
        mtRT.xedges, mtRT.yedges, histMuAny, "Muons (box)", "upper time 10-" + p + "/ns", "lower time 10-" + p + "/ns"
    )
    plotLogHist2d(
        mtRT.xedges,
        mtRT.yedges,
        np.nan_to_num(histLR + 1, posinf=np.max(histLR[np.isfinite(histLR)])),
        "Likelihood ratio (box)",
        "upper time 10-" + p + "/ns",
        "lower time 10-" + p + "/ns",
    )
    # np.nan_to_num(histLR+1,posinf=np.max(histLR[np.isfinite(histLR)]))


# --- start ---
if __name__ == "__main__":
    taggedPmtEvts, primaries = loadData(["data/protonbbww/", "data/gammabbww/"], 0)
    mtLR = MuTagLR().load("models/mu_tag_LR_PEs")
    mtRT = MuTagRise().load("models/mu_tag_LR_rise")
    p = "50"
    diffUpper = taggedPmtEvts["per" + p + "Upper"] - taggedPmtEvts["per10Upper"]
    diffLower = taggedPmtEvts["per" + p + "Lower"] - taggedPmtEvts["per10Lower"]

    """
    # upper cell
    *_, histEOnly, histMuAny, histLR = makeHistograms(mtLR.xedges,mtRT.yedges,taggedPmtEvts,"upper",diffUpper)
    plotLogHist2d(mtLR.xedges,mtRT.yedges,histEOnly,"Electron charge time correlation","upper cell PEs","upper time 10-"+p+"/ns")
    plotLogHist2d(mtLR.xedges,mtRT.yedges,histMuAny,"Muon charge time correlation","upper cell PEs","upper time 10-"+p+"/ns")
    plotLogHist2d(mtLR.xedges,mtRT.yedges,histMuAny+histEOnly,"Charge time correlation","upper cell PEs","upper time 10-"+p+"/ns")
    #plotLogHist2d(mtLR.xedges,mtRT.yedges,histLR,"Charge time likelihood ratio","upper cell PEs","upper time 10-"+p+"/ns")
    # lower cell
    *_, histEOnly, histMuAny, histLR = makeHistograms(mtLR.xedges,mtRT.yedges,taggedPmtEvts,"lower",diffLower)
    plotLogHist2d(mtLR.xedges,mtRT.yedges,histEOnly,"Electron charge time correlation","lower cell PEs","lower time 10-"+p+"/ns")
    plotLogHist2d(mtLR.xedges,mtRT.yedges,histMuAny,"Muon charge time correlation","lower cell PEs","lower time 10-"+p+"/ns")
    plotLogHist2d(mtLR.xedges,mtRT.yedges,histMuAny+histEOnly,"Charge time correlation","lower cell PEs","lower time 10-"+p+"/ns")
    #plotLogHist2d(mtLR.xedges,mtRT.yedges,histLR,"Charge time likelihood ratio","lower cell PEs","lower time 10-"+p+"/ns")
    #"""

    # box cut
    """
    boxX = np.array([40,300])#np.array([20,90])
    boxY = np.array([60,1000])#np.array([30,110])
    boxcut = taggedPmtEvts[np.logical_and(taggedPmtEvts["upper"] > boxX[0],taggedPmtEvts["upper"] < boxX[1])]
    boxcut = boxcut[np.logical_and(boxcut["lower"] > boxY[0], boxcut["lower"] < boxY[1])]
    # mark charge box
    *_, histEOnly, histMuAny, histLR = makeHistograms(mtLR.xedges,mtLR.yedges,taggedPmtEvts)
    plotLogHist2d(mtLR.xedges,mtLR.yedges,histEOnly+histMuAny,"charge histogram")
    plt.gca().add_patch(Rectangle((boxX[0],boxY[0]),boxX[1]-boxX[0],boxY[1]-boxY[0],linewidth=1,edgecolor='r',facecolor='none'))
    plotLogHist2d(mtLR.xedges,mtLR.yedges,histLR,"likelihood ratio")
    plt.gca().add_patch(Rectangle((boxX[0],boxY[0]),boxX[1]-boxX[0],boxY[1]-boxY[0],linewidth=1,edgecolor='r',facecolor='none'))
    # plot times
    plotTimes(mtRT, p, boxcut)
    #"""

    # LR cut
    """
    cut = 2
    lrcut = taggedPmtEvts[mtLR.muonScore(taggedPmtEvts) > cut]
    *_, histEOnly, histMuAny, histLR = makeHistograms(mtLR.xedges,mtLR.yedges,taggedPmtEvts)
    plotLogHist2d(mtLR.xedges,mtLR.yedges,histEOnly+histMuAny,"charge histogram")
    plt.contour(*np.meshgrid(mtLR.xedges[:-1],mtLR.yedges[:-1], indexing="ij"), histLR > cut,levels=[0,1],cmap="Reds_r")
    plotTimes(mtRT,p,lrcut)
    #"""
    # convert to float32
    mtRT.xedges = mtRT.xedges.astype("float32")
    mtRT.yedges = mtRT.yedges.astype("float32")
    # 4x4 charge grid
    xcgrid = np.zeros(5).astype("float32")
    ycgrid = np.zeros(5).astype("float32")
    xcgrid[1:] = np.geomspace(10, 10**4, 4).astype("float32")
    ycgrid[1:] = np.geomspace(10, 10**4, 4).astype("float32")
    *_, histEOnly, histMuAny, histLR = makeHistograms(mtLR.xedges, mtLR.yedges, taggedPmtEvts)
    plt.figure(1)
    plotLogHist2d(mtLR.xedges, mtLR.yedges, histEOnly + histMuAny, "charge histogram", figure=False)
    rtHists = np.zeros((4, 4, mtRT.xedges.shape[0] - 1, mtRT.yedges.shape[0] - 1))
    for i in np.arange(0, 4):
        for j in np.arange(0, 4):
            plt.figure(1)
            plt.gca().add_patch(
                Rectangle(
                    (xcgrid[i], ycgrid[j]),
                    xcgrid[i + 1] - xcgrid[i],
                    ycgrid[j + 1] - ycgrid[j],
                    linewidth=1,
                    edgecolor="r",
                    facecolor="none",
                )
            )
            boxcut = taggedPmtEvts[
                np.logical_and(taggedPmtEvts["upper"] >= xcgrid[i], taggedPmtEvts["upper"] < xcgrid[i + 1])
            ]
            boxcut = boxcut[np.logical_and(boxcut["lower"] >= ycgrid[j], boxcut["lower"] < ycgrid[j + 1])]
            diffUpper = np.clip(boxcut["per" + p + "Upper"] - boxcut["per10Upper"], *mtRT.xedges[[0, -1]])
            diffLower = np.clip(boxcut["per" + p + "Lower"] - boxcut["per10Lower"], *mtRT.yedges[[0, -1]])
            *_, histEOnly, histMuAny, histLR = makeHistograms(mtRT.xedges, mtRT.yedges, boxcut, diffUpper, diffLower)
            rtHists[i, j, :, :] = histLR
            #'''
            plt.figure(2)
            if not np.any(np.isfinite(histLR)):
                continue
            plt.subplot(4, 4, 12 - 4 * j + i + 1)
            # plotLogHist2d(mtRT.xedges,mtRT.yedges,histEOnly+histMuAny,"Time histogram","upper time 10-"+p+"/ns","lower time 10-"+p+"/ns",False)
            # plotLogHist2d(mtRT.xedges,mtRT.yedges,histEOnly,"Electron only events","upper time 10-"+p+"/ns","lower time 10-"+p+"/ns",False)
            # plotLogHist2d(mtRT.xedges,mtRT.yedges,histMuAny,"Muon events","upper time 10-"+p+"/ns","lower time 10-"+p+"/ns",False)
            plotLogHist2d(
                mtRT.xedges,
                mtRT.yedges,
                np.nan_to_num(histLR + 1, posinf=np.max(histLR[np.isfinite(histLR)])),
                "Likelihood ratio ({:n}|{:n})".format(i, j),
                "upper time 10-" + p + "/ns",
                "lower time 10-" + p + "/ns",
                False,
            )
            # plotLogHist2d(mtRT.xedges,mtRT.yedges,histLR,"Likelihood ratio ({:n}|{:n})".format(i,j),"upper time 10-"+p+"/ns","lower time 10-"+p+"/ns",False)
            #'''
    # plt.tight_layout()

    # create tagger
    mtCT = MuTagChargeRise(
        "10", p, mtLR.xedges, mtLR.yedges, mtRT.xedges, mtRT.yedges, xcgrid, ycgrid, mtLR.histLR, rtHists
    )
    mtCT.save("models/mu_tag_LR_CT")

    #'''
    # scoreCT = muonScoreCT(taggedPmtEvts, mtRT.xedges, mtRT.yedges, p, xcgrid, ycgrid, rtHists)
    # scoreCT *= mtLR.muonScore(taggedPmtEvts)
    scoreCT = mtCT.muonScore(taggedPmtEvts)
    # plot ROC curve
    plt.figure()
    _, muAny = getEMuTags(taggedPmtEvts)
    y_pred = np.nan_to_num(scoreCT, nan=float_info.min, neginf=float_info.min, posinf=float_info.max)
    fpr, tpr, cuts = roc_curve(muAny, y_pred)
    plt.plot(fpr, tpr)
    #'''

    plt.show()
