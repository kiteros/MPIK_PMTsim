from muon_tagging import MuTagML
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from shower_analysis import getEMuTags, loadData, plotROC, tagShowers, energyDependentAnalysis
from numpy.lib import recfunctions as rfn
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve
from keras.callbacks import EarlyStopping

# setup net
trainModel = True
filename = "models/mu_tag_ML_at6uwf"
if trainModel:
    inputs = keras.Input(shape=(6,))
    x = layers.Dense(6, activation="relu")(inputs)
    outputs = layers.Dense(1, activation="sigmoid")(x) #TODO add activation (tanh is bad)
    model = keras.Model(inputs=inputs, outputs=outputs, name="mu_tag_ML")
    model.summary()
    model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=False),
        optimizer=keras.optimizers.Adam(learning_rate=0.01),metrics=[keras.metrics.AUC(),keras.metrics.Precision()])
else:
    model = keras.models.load_model(filename)

# load data
paths = ["data/protonbbww/","data/gammabbww/"]
taggedPmtEvts, primaries = loadData(paths,20)
inputs = model.get_layer("input_1").get_config()["batch_input_shape"][1]
if inputs == 8:
    data = taggedPmtEvts[["upper","lower","firstUpper","firstLower","per10Upper","per10Lower","per90Upper","per90Lower"]]
    data = rfn.structured_to_unstructured(data, dtype=float)
elif inputs == 6:
    data = taggedPmtEvts[["upper","lower","per10Upper","per10Lower","per90Upper","per90Lower"]]
    data = rfn.structured_to_unstructured(data, dtype=float)
    for i in np.arange(2,6):
        data[:,i] -= taggedPmtEvts["firstUpper"]
else:
    raise NotImplementedError("Only input shapes 6 and 8 supported.")
_, muAny = getEMuTags(taggedPmtEvts)

# prepare training data
idx = np.arange(data.shape[0])
np.random.default_rng().shuffle(idx)
#TODO how to get good statistics and a reasonable size?
train, test = np.split(idx,[int(data.shape[0]*0.1)])
#train = train[muAny[train]*1 + np.random.random(muAny[train].shape) > 1-muAny[train].sum()/train.size] # sample down
x_train = data[train]
x_test = data[test]
y_train = muAny[train]*1
y_test = muAny[test]*1
print(y_train.sum()/y_train.size)
#weights = {0: y_train.size/(y_train.size-y_train.sum()), 1: y_train.size/y_train.sum()}

# train model
if trainModel:
    history = model.fit(x_train, y_train, batch_size=64, epochs=20, validation_split=0.2, callbacks=[EarlyStopping(patience=3,restore_best_weights=True)])
    model.save(filename)

# test model
'''
test_scores = model.evaluate(x_test, y_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])#'''

# ROC curve
y_pred = model(x_test).numpy().ravel()
fpr, tpr, cuts = roc_curve(y_test, y_pred)
# plot
plt.figure()
plt.title("ROC curve")
plt.plot(fpr,tpr)
plt.xlabel("false muons")
plt.ylabel("true muons")
plt.scatter(fpr[::100],tpr[::100],c=cuts[::100])
plt.colorbar()

# energy dependent analysis
mt = MuTagML(model)
#TODO actually do this on the test data and not everything

# settings
edst = False
if edst:
    plotEdst = True
    cuts = np.array([-1,0,1])
    sep = np.linspace(0,200,200)
    eBinCnt = 4
else:
    plotEdst = False
    cuts = np.linspace(0,1,50)
    sep = np.logspace(-10,0,200) #np.linspace(0,500,500)
    eBinCnt = 4
plotHists = False

# tag showers
cnts, tCnts, ids = tagShowers(mt, taggedPmtEvts, cut=cuts, truth=True, ratio=True, makeIds=True)

# analyse
energyDependentAnalysis(cnts, tCnts, cuts, sep, ids, primaries, plotEdst, eBinCnt)

# show plots
plt.show()
