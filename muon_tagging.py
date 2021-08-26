import numpy as np
from shower_analysis import muonScoreLR
from tensorflow import keras
from numpy.lib import recfunctions as rfn


class MuonTagger:
    """
    Base class for muon taggers.
    Subclasses should implement all defined methods and a default constructor.
    """
    
    def load(self,filename):
        """
        Loads the muon tagger from a file written by `save()`.
        """
        pass

    def save(self,filename):
        """
        Saves the muon tagger to a file.
        """
        pass

    def muonScore(self,taggedPmtEvents):
        """
        Assigns a score to every event in `taggedPmtEvents`.

        Parameters
        ----------
        taggedPmtEvents - structured array
            array containing data from PMT events; required columns depend on used tagger

        Returns
        -------
        ndarray
            array of muon scores; range of scores depends on used tagger
        """
        pass

class MuTagLR(MuonTagger):
    """
    This class tags muons based on the distribution of PEs in the upper and lower chamber.
    Required columns for score calculation: upper, lower
    """

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
    """
    This class tags muons based on the distribution of rise times in the upper and lower chamber.
    Required columns for score calculation: per90Upper, per10Upper, per90Lower, per10Lower
    """

    def muonScore(self, taggedPmtEvents):
        return muonScoreLR(self.xedges, self.yedges, taggedPmtEvents["per90Upper"]-taggedPmtEvents["per10Upper"],
            taggedPmtEvents["per90Lower"]-taggedPmtEvents["per10Lower"], self.histLR)

class MuTagML(MuonTagger):
    """
    This class tags muons based on machine learning techniques.
    Required columns for score calculation: upper, lower, firstUpper, firstLower, per10Upper, per10Lower, per90Upper, per90Lower
    """    

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
    """
    This class adds the muon scores of two or more `MuonTagger`s.
    Required columns depend on given taggers. Cannot be saved.
    """

    def __init__(self, taggers):
        self.taggers = taggers
    
    def muonScore(self, taggedPmtEvents):
        return np.sum([mt.muonScore(taggedPmtEvents) for mt in self.taggers],axis=0)

class ProductTagger(MuonTagger):
    """
    This class multiplies the muon scores of two or more `MuonTagger`s.
    Required columns depend on given taggers. Cannot be saved.
    """

    def __init__(self, taggers):
        self.taggers = taggers
    
    def muonScore(self, taggedPmtEvents):
        return np.product([mt.muonScore(taggedPmtEvents) for mt in self.taggers],axis=0)

