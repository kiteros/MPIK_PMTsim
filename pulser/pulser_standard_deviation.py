#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import rv_histogram, randint, poisson, expon, exponnorm
from scipy.signal import resample
import scipy.integrate as integrate

import sys
sys.path.insert(0, '/home/jebach/Documents/flashcam/pmt-trace-simulation-master/PMTtraceSIM_draft/debug_fcts')
sys.path.insert(0, '/home/jebach/Documents/flashcam/pmt-trace-simulation-master/PMTtraceSIM_draft/simulation')
sys.path.insert(0, '/home/jebach/Documents/flashcam/pmt-trace-simulation-master/PMTtraceSIM_draft/darkcounts')
sys.path.insert(0, '/home/jebach/Documents/flashcam/pmt-trace-simulation-master/PMTtraceSIM_draft/baselineshift')

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

from scipy import signal


gain = 10

esim = TraceSimulation(
    ampSpec="../data/spe_R11920-RM_ap0.0002.dat",
    timeSpec="../data/bb3_1700v_timing.txt",
    #pulseShape="data/pulse_FlashCam_7dynode_v2a.dat",
    background_rate = 1e6,
    gain=gain,
    no_signal_duration = 1e4,
    noise=0,
)

plt.figure()
plt.plot(*esim.ampSpec)
print(esim.ampStddev)


pulse = Pulser(step=esim.t_step, duration=esim.no_signal_duration, pulse_type="pulsed")
evts = pulse.generate_all()

evts_br, k_evts = esim.simulateBackground(evts)

# pmt signal
times, pmtSig, uncertainty_pmt = esim.simulatePMTSignal(evts_br, k_evts) #TODO : make uncertainty from the simulatePMTSignal, with ampdist.rvs(). For now sufficient


eleSig, uncertainty_ele = esim.simulateElectronics(pmtSig, uncertainty_pmt, times)


# adc signal
stimes, samples, samples_unpro, uncertainty_sampled = esim.simulateADC(times, eleSig, uncertainty_ele, 1)



###Now the goal is to identify every peak

maxs, _ = signal.find_peaks(samples, prominence=1) ###promeminence devrait dependre du gain ? Non
max_values_stimes = stimes[maxs]
max_values = samples[maxs]


plt.figure()
plt.plot(stimes, samples)
plt.scatter(max_values_stimes, max_values, marker='o', color='red')
plt.xlabel("Number of samples")
plt.ylabel("LSB")



###Do an histogram of the peaks
hist, bins = np.histogram(max_values, density=True, bins=30)
width = (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2


plt.figure()
plt.bar(center, hist, align='center', width=width)
plt.title("gain : " + str(10))
plt.show()


print(np.std(max_values))
#print(esim.ampStddev*gain*np.sqrt(10))
print(np.sqrt(pulse.pe_intensity*gain**2*esim.ampStddev**2+pulse.pe_intensity*gain**2))