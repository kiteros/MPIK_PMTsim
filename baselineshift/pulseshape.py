#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm, rv_histogram, randint, poisson, expon, exponnorm, skew, gamma
from scipy.signal import resample

import sys
sys.path.insert(0, '/home/jebach/Documents/flashcam/pmt-trace-simulation-master/PMTtraceSIM_draft/debug_fcts')
sys.path.insert(0, '/home/jebach/Documents/flashcam/pmt-trace-simulation-master/PMTtraceSIM_draft/simulation')
sys.path.insert(0, '/home/jebach/Documents/flashcam/pmt-trace-simulation-master/PMTtraceSIM_draft/darkcounts')

from pulser import Pulser
import statistics
import scipy.integrate as integrate
import math 
import sys
from scipy.optimize import curve_fit
import scipy.special as sse
import scipy

"""
Allows to print the current pulse shape
"""

ps = np.loadtxt("../data/pulse_FlashCam_7dynode_v2a.dat", unpack=True)

plt.figure()
plt.plot(ps[1])
plt.xlabel("Samples")
plt.ylabel("Relative amplitude")
plt.show()