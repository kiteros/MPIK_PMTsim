from muon_tagging import MuTagLR, MuTagRise
import numpy as np
from matplotlib import pyplot as plt
from shower_analysis import *

taggedPmtEvts, primaries = loadData(["data/protonbbww/","data/gammabbww/"],0)
mtLR = MuTagLR().load("models/mu_tag_LR_PEs")
mtRT = MuTagRise().load("models/mu_tag_LR_rise")
p = "20"
diffUpper = taggedPmtEvts["per"+p+"Upper"]-taggedPmtEvts["per10Upper"]
diffLower = taggedPmtEvts["per"+p+"Lower"]-taggedPmtEvts["per10Lower"]

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

plt.show()
