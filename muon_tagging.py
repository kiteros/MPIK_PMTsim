import numpy as np
from shower_analysis import muonScoreLR
from tensorflow import keras
from numpy.lib import recfunctions as rfn


class MuonTagger:
    
    def load(self,filename):
        pass

    def save(self,filename):
        pass

    def muonScore(self,taggedPmtEvents):
        pass

class MuTagLR(MuonTagger):

    def __init__(self, xedges=None, yedges=None, histLR=None):
        self.xedges = xedges
        self.yedges = yedges
        self.histLR = histLR
    
    def load(self, filename):
        self.xedges = np.load(filename+"/xedges.npy")
        self.yedges = np.load(filename+"/yedges.npy")
        self.histLR = np.load(filename+"/histLR.npy")
        return self
    
    def save(self, filename):
        #TODO mkdir
        np.save(filename+"/xedges.npy",self.xedges)
        np.save(filename+"/yedges.npy",self.yedges)
        np.save(filename+"/histLR.npy",self.histLR)
    
    def muonScore(self, taggedPmtEvents):
        return muonScoreLR(self.xedges, self.yedges, taggedPmtEvents["upper"], taggedPmtEvents["lower"], self.histLR)

class MuTagRise(MuTagLR):

    def muonScore(self, taggedPmtEvents):
        return muonScoreLR(self.xedges, self.yedges, taggedPmtEvents["per90Upper"]-taggedPmtEvents["per10Upper"],
            taggedPmtEvents["per90Lower"]-taggedPmtEvents["per10Lower"], self.histLR)

class MuTagML(MuonTagger):

    def __init__(self, model=None):
        self.model = model
    
    def load(self, filename):
        self.model = keras.models.load_model("models/mu_tag_ML")
        return self
    
    def save(self, filename):
        self.model.save(filename)
    
    def muonScore(self, taggedPmtEvts):
        inputs = self.model.get_layer("input_1").get_config()["batch_input_shape"][1]
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
        return self.model(data).numpy().ravel()

class SumTagger(MuonTagger):

    def __init__(self, taggers):
        self.taggers = taggers
    
    def muonScore(self, taggedPmtEvents):
        return np.sum([mt.muonScore(taggedPmtEvents) for mt in self.taggers],axis=0)

