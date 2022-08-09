#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import rv_histogram, randint, poisson, expon, exponnorm
from scipy.signal import resample
import scipy.integrate as integrate

import sys
import os
#sys.path.insert(0, '/home/jebach/Documents/flashcam/pmt-trace-simulation-master/PMTtraceSIM_draft/debug_fcts')
#sys.path.insert(0, '/home/jebach/Documents/flashcam/pmt-trace-simulation-master/PMTtraceSIM_draft/simulation')
#sys.path.insert(0, '/home/jebach/Documents/flashcam/pmt-trace-simulation-master/PMTtraceSIM_draft/darkcounts')
#sys.path.insert(0, '/home/jebach/Documents/flashcam/pmt-trace-simulation-master/PMTtraceSIM_draft/baselineshift')

print(__file__)
p1 = os.path.abspath(__file__+"/../../")
sys.path.insert(0, p1+"\\debug_fcts")
sys.path.insert(0, p1+"\\simulation")
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

from scipy import signal
import scipy.special as sse

import matplotlib.patches as patches

esim = TraceSimulation(
    #ampSpec="../data/spe_R11920-RM_ap0.0002.dat",
    timeSpec="../data/bb3_1700v_timing.txt",
    #pulseShape="data/pulse_FlashCam_7dynode_v2a.dat",
    background_rate = 0,
    gain=10,
    no_signal_duration = 1e6,
    noise=0,
    oversamp=40
)

plt.figure()
plt.plot(*esim.ampSpec)

plt.figure()
plt.plot(*esim.pulseShape)


pulse = Pulser(step=esim.t_step,freq=5e6, duration=esim.no_signal_duration, pulse_type="pulsed")
evts = pulse.generate_all()

#evts = np.array([1000])


evts_br, k_evts = esim.simulateBackground(evts)

# pmt signal
times, pmtSig, uncertainty_pmt = esim.simulatePMTSignal(evts_br, k_evts) #TODO : make uncertainty from the simulatePMTSignal, with ampdist.rvs(). For now sufficient


eleSig, uncertainty_ele = esim.simulateElectronics(pmtSig, uncertainty_pmt, times)



# adc signal
stimes, samples, samples_unpro, uncertainty_sampled = esim.simulateADC(times, eleSig, uncertainty_ele, 1)

plt.figure()
plt.plot(times, pmtSig)
plt.title("pmtSig")


plt.figure()
plt.plot(times, eleSig)
plt.title("eleSig")

plt.figure()
plt.plot(stimes, samples)
plt.title("samples")


print(esim.t_step)
print(esim.pulse_time_step)

maxs, _ = signal.find_peaks(pmtSig, prominence=0.5) ###promeminence devrait dependre du gain ? Non
max_values_stimes = times[maxs]
max_values = pmtSig[maxs]

###Do an histogram of the peaks
hist, bins = np.histogram(max_values, density=True, bins=30)
width = (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2



plt.figure()
plt.bar(center, hist, align='center', width=width)
plt.title("ampdist")

period = (1.0/pulse.max_frequency)*(1e9) #In ns

max_values_stimes = [x % period for x in max_values_stimes]
###Do an histogram of the peaks
hist, bins = np.histogram(max_values_stimes, density=True, bins=30)
width = (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2



plt.figure()
plt.bar(center, hist, align='center', width=width)
plt.title("time dist")

print("time std", np.std(max_values_stimes))
print("pulser time std", pulse.pulse_std)


####do the same for elegsig

maxs, _ = signal.find_peaks(eleSig, prominence=0.5) ###promeminence devrait dependre du gain ? Non
max_values_stimes = times[maxs]
max_values = eleSig[maxs]

hist, bins = np.histogram(max_values, density=True, bins=30)
width = (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2

plt.figure()
plt.bar(center, hist, align='center', width=width)
plt.title("eleSig")

plt.show()