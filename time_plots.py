from sys import float_info
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
'''
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

# 4x4 charge grid
xcgrid = np.zeros(5)
ycgrid = np.zeros(5)
xcgrid[1:] = np.geomspace(10,10**4,4)
ycgrid[1:] = np.geomspace(10,10**4,4)
*_, histEOnly, histMuAny, histLR = makeHistograms(mtLR.xedges,mtLR.yedges,taggedPmtEvts)
plt.figure(1)
plotLogHist2d(mtLR.xedges,mtLR.yedges,histEOnly+histMuAny,"charge histogram",figure=False)
rtHists = np.zeros((4,4,mtRT.xedges.shape[0]-1,mtRT.yedges.shape[0]-1))
for i in np.arange(0,4):
    for j in np.arange(0,4):
        plt.figure(1)
        plt.gca().add_patch(Rectangle((xcgrid[i],ycgrid[j]),xcgrid[i+1]-xcgrid[i],ycgrid[j+1]-ycgrid[j],linewidth=1,edgecolor='r',facecolor='none'))
        boxcut = taggedPmtEvts[np.logical_and(taggedPmtEvts["upper"] >= xcgrid[i],taggedPmtEvts["upper"] < xcgrid[i+1])]
        boxcut = boxcut[np.logical_and(boxcut["lower"] >= ycgrid[j], boxcut["lower"] < ycgrid[j+1])]
        diffUpper = boxcut["per"+p+"Upper"]-boxcut["per10Upper"]
        diffLower = boxcut["per"+p+"Lower"]-boxcut["per10Lower"]
        *_, histEOnly, histMuAny, histLR = makeHistograms(mtRT.xedges,mtRT.yedges,boxcut,diffUpper,diffLower)
        rtHists[i,j,:,:] = histLR
        '''
        plt.figure(2)
        if not np.any(np.isfinite(histLR)): continue
        plt.subplot(4,4,12-4*j+i+1)
        plotLogHist2d(mtRT.xedges,mtRT.yedges,histEOnly+histMuAny,"Time histogram","upper time 10-"+p+"/ns","lower time 10-"+p+"/ns",False)
        #plotLogHist2d(mtRT.xedges,mtRT.yedges,np.nan_to_num(histLR+1,posinf=np.max(histLR[np.isfinite(histLR)])),"Likelihood ratio ({:n}|{:n})".format(i,j),"upper time 10-"+p+"/ns","lower time 10-"+p+"/ns",False)
        #'''
#plt.tight_layout()


diffUpper = taggedPmtEvts["per"+p+"Upper"]-taggedPmtEvts["per10Upper"]
diffLower = taggedPmtEvts["per"+p+"Lower"]-taggedPmtEvts["per10Lower"]
# digitize charge and time
#TODO fix
uppIdxC = np.digitize(np.clip(taggedPmtEvts["upper"],*xcgrid[[0,-1]]), xcgrid)-1
lowIdxC = np.digitize(np.clip(taggedPmtEvts["lower"],*ycgrid[[0,-1]]), ycgrid)-1
uppIdxT = np.digitize(np.clip(diffUpper,*mtRT.xedges[[0,-1]]), mtRT.xedges)-1
lowIdxT = np.digitize(np.clip(diffLower,*mtRT.yedges[[0,-1]]), mtRT.yedges)-1
scoreCT = rtHists[uppIdxC,lowIdxC,uppIdxT,lowIdxT]
sctt = scoreCT[~np.isnan(scoreCT)]

plt.show()
