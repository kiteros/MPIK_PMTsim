import numpy as np
from shower_analysis import muonScoreLR
from numpy.lib import recfunctions as rfn
from pathlib import Path
from time_plots import muonScoreCT
import json


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

    def __init__(self, xedges=None, yedges=None, histLR=None, config=None):
        self.xedges = xedges
        self.yedges = yedges
        self.histLR = histLR
        if config==None:
            config = {"type" : "MuTagLR"}
        self.config = config
    
    def load(self, filename):
        self.xedges = np.load(filename+"/xedges.npy")
        self.yedges = np.load(filename+"/yedges.npy")
        self.histLR = np.load(filename+"/histLR.npy")
        with open(filename+"/config.json","r") as f:
            self.config = json.load(f)
        return self
    
    def save(self, filename):
        Path(filename).mkdir(exist_ok=True)
        np.save(filename+"/xedges.npy",self.xedges)
        np.save(filename+"/yedges.npy",self.yedges)
        np.save(filename+"/histLR.npy",self.histLR)
        with open(filename+"/config.json","w") as f:
            json.dump(self.config,f)
    
    def muonScore(self, taggedPmtEvents):
        return muonScoreLR(self.xedges, self.yedges, taggedPmtEvents["upper"], taggedPmtEvents["lower"], self.histLR)

class MuTagRise(MuTagLR):
    """
    This class tags muons based on the distribution of rise times in the upper and lower chamber.
    Required columns for score calculation: per90Upper, per10Upper, per90Lower, per10Lower
    """

    def __init__(self, xedges=None, yedges=None, histLR=None, config=None, low="10", high="50"):
        super().__init__(xedges=xedges, yedges=yedges, histLR=histLR, config=config)
        self.config["type"] = "MuTagRise"
        if not "low" in self.config:
            self.config["low"] = low
        if not "high" in self.config:
            self.config["high"] = high

    def muonScore(self, taggedPmtEvents):
        return muonScoreLR(self.xedges, self.yedges, taggedPmtEvents["per"+self.config["high"]+"Upper"]-taggedPmtEvents["per"+self.config["low"]+"Upper"],
            taggedPmtEvents["per"+self.config["high"]+"Lower"]-taggedPmtEvents["per"+self.config["low"]+"Lower"], self.histLR)

class MuTagML(MuonTagger):
    """
    This class tags muons based on machine learning techniques.
    Required columns for score calculation: upper, lower, firstUpper, firstLower, per10Upper, per10Lower, per90Upper, per90Lower
    """    

    def __init__(self, model=None, config=None):
        self.model = model
        self.config = config
    
    def load(self, filename):
        from tensorflow import keras
        self.model = keras.models.load_model(filename)
        with open(filename+"/config.json","r") as f:
            self.config = json.load(f)
        return self
    
    def save(self, filename):
        self.model.save(filename)
        with open(filename+"/config.json","w") as f:
            json.dump(self.config,f)
    
    def muonScore(self, taggedPmtEvts):
        fields = self.config["fields"]
        data = taggedPmtEvts[fields]
        if "subtract" in self.config:
            for f,s in zip(self.config["subtract_from"],self.config["subtract"]):
                data[f] -= taggedPmtEvts[s]
        data = rfn.structured_to_unstructured(data, dtype=float)
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

class MuTagChargeRise(MuonTagger):
    """
    This class tags muons based on the distribution of PEs and rise times in the upper and lower chamber.
    Required columns for score calculation: upper, lower, perXUpper, per10Upper, perXLower, per10Lower (X may vary)
    """

    def __init__(self, low=None, high=None, xcedges=None, ycedges=None, xtedges=None, ytedges=None, xcgrid=None, ycgrid=None, histLR=None, rtHists=None):
        self.xtedges = xtedges
        self.ytedges = ytedges
        self.xcedges = xcedges
        self.ycedges = ycedges
        self.xcgrid = xcgrid
        self.ycgrid = ycgrid
        self.histLR = histLR
        self.rtHists = rtHists
        self.high = high
        self.low = low
    
    def load(self, filename):
        self.xcedges = np.load(filename+"/xcedges.npy")
        self.ycedges = np.load(filename+"/ycedges.npy")
        self.xtedges = np.load(filename+"/xtedges.npy")
        self.ytedges = np.load(filename+"/ytedges.npy")
        self.xcgrid = np.load(filename+"/xcgrid.npy")
        self.ycgrid = np.load(filename+"/ycgrid.npy")
        self.histLR = np.load(filename+"/histLR.npy")
        self.rtHists = np.load(filename+"/rtHists.npy")
        with open(filename+"/config.json","r") as f:
            config = json.load(f)
        self.low = config["low"]
        self.high = config["high"]
        return self
    
    def save(self, filename):
        Path(filename).mkdir(exist_ok=True)
        np.save(filename+"/xcedges.npy",self.xcedges)
        np.save(filename+"/ycedges.npy",self.ycedges)
        np.save(filename+"/xtedges.npy",self.xtedges)
        np.save(filename+"/ytedges.npy",self.ytedges)
        np.save(filename+"/xcgrid.npy",self.xcgrid)
        np.save(filename+"/ycgrid.npy",self.ycgrid)
        np.save(filename+"/histLR.npy",self.histLR)
        np.save(filename+"/rtHists.npy",self.rtHists)
        conf = {"low":self.low,"high":self.high,"type":"MuTagChargeRise"}
        with open(filename+"/config.json","w") as f:
            json.dump(conf,f)
    
    def muonScore(self, taggedPmtEvents):
        scoreCT = muonScoreCT(taggedPmtEvents, self.xtedges, self.ytedges, self.low, self.high, self.xcgrid, self.ycgrid, self.rtHists)
        scoreCT *= muonScoreLR(self.xcedges, self.ycedges, taggedPmtEvents["upper"], taggedPmtEvents["lower"], self.histLR)
        return scoreCT

