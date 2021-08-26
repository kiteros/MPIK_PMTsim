from muon_tagging import MuTagLR, MuTagRise
import numpy as np
from matplotlib import pyplot as plt
from shower_analysis import *
from matplotlib.patches import Rectangle

def plotTimes(mtRT, p, boxcut):
    diffUpper = boxcut["per"+p+"Upper"]-boxcut["per10Upper"]
    diffLower = boxcut["per"+p+"Lower"]-boxcut["per10Lower"]
    *_, histEOnly, histMuAny, histLR = makeHistograms(mtRT.xedges,mtRT.yedges,boxcut,diffUpper,diffLower)
    plotLogHist2d(mtRT.xedges,mtRT.yedges,histEOnly,"Electron only (box)","upper time 10-"+p+"/ns","lower time 10-"+p+"/ns")
    plotLogHist2d(mtRT.xedges,mtRT.yedges,histMuAny,"Muons (box)","upper time 10-"+p+"/ns","lower time 10-"+p+"/ns")
    plotLogHist2d(mtRT.xedges,mtRT.yedges,np.nan_to_num(histLR+1,posinf=np.max(histLR[np.isfinite(histLR)])),"Likelihood ratio (box)","upper time 10-"+p+"/ns","lower time 10-"+p+"/ns")
    #np.nan_to_num(histLR+1,posinf=np.max(histLR[np.isfinite(histLR)]))

taggedPmtEvts, primaries = loadData(["data/protonbbww/","data/gammabbww/"],0)
mtLR = MuTagLR().load("models/mu_tag_LR_PEs")
mtRT = MuTagRise().load("models/mu_tag_LR_rise")
p = "50"
diffUpper = taggedPmtEvts["per"+p+"Upper"]-taggedPmtEvts["per10Upper"]
diffLower = taggedPmtEvts["per"+p+"Lower"]-taggedPmtEvts["per10Lower"]

'''
# upper cell
*_, histEOnly, histMuAny, histLR = makeHistograms(mtLR.xedges,mtRT.yedges,taggedPmtEvts,"upper",diffUpper)
plotLogHist2d(mtLR.xedges,mtRT.yedges,histEOnly,"Electron charge time correlation","upper cell PEs","upper time 10-"+p+"/ns")
plotLogHist2d(mtLR.xedges,mtRT.yedges,histMuAny,"Muon charge time correlation","upper cell PEs","upper time 10-"+p+"/ns")
plotLogHist2d(mtLR.xedges,mtRT.yedges,histLR,"Charge time likelihood ratio","upper cell PEs","upper time 10-"+p+"/ns")
# lower cell
*_, histEOnly, histMuAny, histLR = makeHistograms(mtLR.xedges,mtRT.yedges,taggedPmtEvts,"lower",diffLower)
plotLogHist2d(mtLR.xedges,mtRT.yedges,histEOnly,"Electron charge time correlation","lower cell PEs","lower time 10-"+p+"/ns")
plotLogHist2d(mtLR.xedges,mtRT.yedges,histMuAny,"Muon charge time correlation","lower cell PEs","lower time 10-"+p+"/ns")
plotLogHist2d(mtLR.xedges,mtRT.yedges,histLR,"Charge time likelihood ratio","lower cell PEs","lower time 10-"+p+"/ns")
#'''

# box cut
#'''
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
#'''

# LR cut
'''
cut = 2
lrcut = taggedPmtEvts[mtLR.muonScore(taggedPmtEvts) > cut]
*_, histEOnly, histMuAny, histLR = makeHistograms(mtLR.xedges,mtLR.yedges,taggedPmtEvts)
plotLogHist2d(mtLR.xedges,mtLR.yedges,histEOnly+histMuAny,"charge histogram")
plt.contour(*np.meshgrid(mtLR.xedges[:-1],mtLR.yedges[:-1], indexing="ij"), histLR > cut,levels=[0,1],cmap="Reds_r")
plotTimes(mtRT,p,lrcut)
#'''

plt.show()
