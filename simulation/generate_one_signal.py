#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import rv_histogram, randint, poisson, expon, exponnorm
from scipy.signal import resample
import scipy.integrate as integrate


import sys
import os

"""
sys.path.insert(0, '/home/jebach/Documents/flashcam/pmt-trace-simulation-master/PMTtraceSIM_draft/debug_fcts')
sys.path.insert(0, '/home/jebach/Documents/flashcam/pmt-trace-simulation-master/PMTtraceSIM_draft/baselineshift')
sys.path.insert(0, '/home/jebach/Documents/flashcam/pmt-trace-simulation-master/PMTtraceSIM_draft/darkcounts')


"""
print(__file__)
p1 = os.path.abspath(__file__+"/../../")
sys.path.insert(0, p1+"\\debug_fcts")
sys.path.insert(0, p1+"\\pulser")
sys.path.insert(0, p1+"\\darkcounts")
sys.path.insert(0, p1+"\\baselineshift")



from pulser import Pulser
import scipy
import math 
from trace_simulation import TraceSimulation
from scipy.optimize import curve_fit
from scipy import odr
from pylab import *
import statistics
import os.path

from bl_shift import BL_shift
from bl_stddev import BL_stddev
from under_c import Under_c
from debug import Debug
#from debug_fcts.baseline import Baseline
from pulse import Pulse


from scipy.stats import norm

from mpl_toolkits.mplot3d import axes3d, Axes3D
from sklearn import linear_model 

from calculate_gains import GainCalculator
import csv


# importing pandas package
import pandas as pd

esim_init = TraceSimulation(
    ampSpec="../data/spe_R11920-RM_ap0.0002.dat",
    timeSpec="../data/bb3_1700v_timing.txt",
    pulseShape="../data/pulse_FlashCam_7dynode_v2a.dat",
    background_rate = 1e8,
    gain=10,
    no_signal_duration = 1e4,

)


pulse = Pulser(step=esim_init.t_step, pulse_type="none")
evts = pulse.generate_all()

esim = TraceSimulation(
    ampSpec="../data/spe_R11920-RM_ap0.0002.dat",
    timeSpec="../data/bb3_1700v_timing.txt",
    #pulseShape="../data/pulse_FlashCam_7dynode_v2a.dat",
    background_rate = 1e8,
    gain=10,
    no_signal_duration = 1e4,
    noise=0.8,
)

plt.figure()
plt.plot(*esim.ampSpec)
plt.grid()
plt.xlabel("Amplitude")
plt.ylabel("Relative probability")

plt.figure()
plt.plot(*esim.pulseShape)
plt.grid()
plt.xlabel("Time [ns]")
plt.ylabel("Relative amplitude")

plt.show()

evts_br, k_evts = esim.simulateBackground(evts)

# pmt signal
times, pmtSig, uncertainty_pmt = esim.simulatePMTSignal(evts_br, k_evts) #TODO : make uncertainty from the simulatePMTSignal, with ampdist.rvs(). For now sufficient


eleSig, uncertainty_ele = esim.simulateElectronics(pmtSig, uncertainty_pmt, times)

plt.figure()
plt.plot(times, eleSig)
plt.show()

# adc signal
stimes, samples, samples_unpro, uncertainty_sampled = esim.simulateADC(times, eleSig, uncertainty_ele, 1)

plt.figure()
plt.plot(stimes, samples, label="gain=10")

plt.xlabel("Time [ns]")
plt.ylabel("LSB")
plt.legend(loc="upper right")




plt.show()


