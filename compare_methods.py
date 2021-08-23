from sys import float_info
from muon_tagging import *
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve
from shower_analysis import *
from mu_tag_LR import energyDependentAnalysis

# load muon taggers
mtLR = MuTagLR().load("models/mu_tag_LR_PEs")
mtRise = MuTagRise().load("models/mu_tag_LR_rise")
mtML = MuTagML().load("models/mu_tag_ML_at6uw")
mtSum = SumTagger([mtRise, mtLR])

# load data
gammas = "data/gammabbww/"
protons = "data/protonbbww/"
taggedPmtEvts = np.load(protons+"taggedPmtEvts2.npy")
taggedPmtEvts = taggedPmtEvts[taggedPmtEvts["distance"] > 20*100]
_, muAny = getEMuTags(taggedPmtEvts)

# plot roc curves
plt.figure()
plt.title("ROC curve")
plt.xlabel("false muons")
plt.ylabel("true muons")
for mt, lbl in zip([mtLR, mtRise, mtML, mtSum],["charge LR", "rise time LR", "charge rise ML", "charge rise LR"]):
    y_pred = mt.muonScore(taggedPmtEvts)
    y_pred = np.nan_to_num(y_pred, nan=float_info.min, neginf=float_info.min, posinf=float_info.max)
    fpr, tpr, cuts = roc_curve(muAny, y_pred)
    plt.plot(fpr,tpr,label=lbl)

plt.legend()

plt.show()
