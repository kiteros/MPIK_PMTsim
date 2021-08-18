import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from shower_analysis import getEMuTags, plotROC
from numpy.lib import recfunctions as rfn
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve

# setup net
trainModel = False
if trainModel:
    inputs = keras.Input(shape=(8,))
    x = layers.Dense(8, activation="relu")(inputs)
    outputs = layers.Dense(1, activation=None)(x) #TODO add activation (tanh?)
    model = keras.Model(inputs=inputs, outputs=outputs, name="mu_tag_ML")
    model.summary()
    model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(learning_rate=0.01),metrics=["accuracy"])
else:
    model = keras.models.load_model("models/mu_tag_ML")

# load data
path = "data/protonbbww/"
taggedPmtEvts = np.load(path+"taggedPmtEvts2.npy")
taggedPmtEvts = taggedPmtEvts[taggedPmtEvts["distance"] > 20*100]
data = taggedPmtEvts[["upper","lower","firstUpper","firstLower","per10Upper","per10Lower","per90Upper","per90Lower"]]
data = rfn.structured_to_unstructured(data, dtype=float)
_, muAny = getEMuTags(taggedPmtEvts)

# prepare training data
idx = np.arange(data.shape[0])
np.random.default_rng().shuffle(idx)
train, test = np.split(idx,[int(data.shape[0]*0.9)])
x_train = data[train]
x_test = data[test]
y_train = muAny[train]*1
y_test = muAny[test]*1

# train model
if trainModel:
    history = model.fit(x_train, y_train, batch_size=64, epochs=2, validation_split=0.2)
    model.save("models/mu_tag_ML")

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

plt.show()
