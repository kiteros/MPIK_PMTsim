import numpy as np
from matplotlib import pyplot as plt
from shower_analysis import *
from muon_tagging import MuTagLR

taggedPmtEvts, primaries = loadData(["data/protonbbww/","data/gammabbww/"],0)
mt = MuTagLR().load("models/mu_tag_LR_PEs")
histMax = mt.histLR > 40
plotLogHist2d(mt.xedges,mt.yedges,histMax)
plt.show()
