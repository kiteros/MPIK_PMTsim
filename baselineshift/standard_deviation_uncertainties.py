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
import scipy.fftpack

from scipy import special



esim_init = TraceSimulation(
    ampSpec="../data/spe_R11920-RM_ap0.0002.dat",
    timeSpec="../data/bb3_1700v_timing.txt",
    pulseShape="../data/pulse_FlashCam_7dynode_v2a.dat",
    background_rate = 3e9,
    gain=10,
    no_signal_duration = 1e4,

)


pulse = Pulser(step=esim_init.t_step, pulse_type="none")
evts = pulse.generate_all()



smp = []



#############jiggle of the signal (G=1, N=0)

"""

for times in range(30):

	###generate n times the same signal with the same parameters, and measure the standard deviaiton of the sample

    esim = TraceSimulation(
        ampSpec="data/spe_R11920-RM_ap0.0002.dat",
        timeSpec="data/bb3_1700v_timing.txt",
        pulseShape="data/pulse_FlashCam_7dynode_v2a.dat",
        background_rate = 1e9,
        gain=1,
        no_signal_duration = 1e3,
        noise=0,
    )

    evts_br, k_evts = esim.simulateBackground(evts)

    # pmt signal
    times, pmtSig, uncertainty_pmt = esim.simulatePMTSignal(evts_br, k_evts) #TODO : make uncertainty from the simulatePMTSignal, with ampdist.rvs(). For now sufficient


    eleSig, uncertainty_ele = esim.simulateElectronics(pmtSig, uncertainty_pmt, times)

    # adc signal
    stimes, samples, samples_unpro, uncertainty_sampled = esim.simulateADC(times, eleSig, uncertainty_ele, 1)

    smp.append(samples[len(samples)//2])


plt.figure()
plt.plot(np.linspace(0,1,len(smp)), smp)
plt.show()
"""

############### poisson stuff

"""
lamda = esim_init.lamda
pmtSig_smp = []

for times in range(300):

	###generate n times the same signal with the same parameters, and measure the standard deviaiton of the sample

    esim = TraceSimulation(
        ampSpec="data/spe_R11920-RM_ap0.0002.dat",
        timeSpec="data/bb3_1700v_timing.txt",
        pulseShape="data/pulse_FlashCam_7dynode_v2a.dat",
        background_rate = 1e9,
        gain=1,
        no_signal_duration = 1e3,
        noise=0,
    )

    evts_br, k_evts = esim.simulateBackground(evts)

    # pmt signal
    times, pmtSig, uncertainty_pmt = esim.simulatePMTSignal(evts_br, k_evts) #TODO : make uncertainty from the simulatePMTSignal, with ampdist.rvs(). For now sufficient


    eleSig, uncertainty_ele = esim.simulateElectronics(pmtSig, uncertainty_pmt, times)

    # adc signal
    stimes, samples, samples_unpro, uncertainty_sampled = esim.simulateADC(times, eleSig, uncertainty_ele, 1)

    pmtSig_smp.append(eleSig[len(eleSig)//2])


sigma_mean = np.std(pmtSig_smp)
sigma_sample = np.sqrt(len(pmtSig_smp))*sigma_mean

mean = np.mean(pmtSig_smp)


print(esim_init.ampStddev*lamda+esim_init.ampDist_drift*np.sqrt(lamda))

plt.figure()
plt.plot(np.linspace(0,1,len(pmtSig_smp)), pmtSig_smp)
plt.plot(np.linspace(0,1,len(pmtSig_smp)), np.repeat(mean, len(pmtSig_smp)))
plt.plot(np.linspace(0,1,len(pmtSig_smp)), np.repeat(mean+sigma_mean, len(pmtSig_smp)))
plt.plot(np.linspace(0,1,len(pmtSig_smp)), np.repeat(mean-sigma_mean, len(pmtSig_smp)))
plt.show()
"""

#############
#crosscheck the average number of events follows a poisson distrib

"""

def generate_PMT_visualisation(evts_br, k_evts,  esim):
	x = np.linspace(evts_br.min(), evts_br.max(), num=int((evts_br.max()-evts_br.min())//esim.t_step))
	y = np.zeros(x.shape)

	for i in range(len(evts_br)):
		for j in range(len(x)):

			if evts_br[i] < x[j] + esim.t_step/2 and evts_br[i] > x[j] - esim.t_step/2:
				y[j]= k_evts[i]

	return x, y

poisson_events = []


for times in range(1):

	###generate n times the same signal with the same parameters, and measure the standard deviaiton of the sample

    esim = TraceSimulation(
        ampSpec="data/spe_R11920-RM_ap0.0002.dat",
        timeSpec="data/bb3_1700v_timing.txt",
        pulseShape="data/pulse_FlashCam_7dynode_v2a.dat",
        background_rate = 1e10,
        gain=1,
        no_signal_duration = 1e2,
        noise=0,
    )

    evts_br, k_evts = esim.simulateBackground(evts)

    # pmt signal
    times, pmtSig, uncertainty_pmt = esim.simulatePMTSignal(evts_br, k_evts) #TODO : make uncertainty from the simulatePMTSignal, with ampdist.rvs(). For now sufficient


    eleSig, uncertainty_ele = esim.simulateElectronics(pmtSig, uncertainty_pmt, times)

    # adc signal
    stimes, samples, samples_unpro, uncertainty_sampled = esim.simulateADC(times, eleSig, uncertainty_ele, 1)

    plt.figure()
    x, y = generate_PMT_visualisation(evts_br, k_evts, esim)

    lamda = esim.lamda

    mean = np.mean(y)
    stddev = np.std(y)

    print("THEOREICAL -- Background events")
    print("MEAN", lamda)
    print("STD", np.sqrt(lamda))
    print("SIZE", len(y))
    plt.plot(x,y)
    plt.plot(x, np.repeat(mean, len(y)))


    plt.plot(x, np.repeat(mean+stddev, len(y)))
    plt.plot(x, np.repeat(mean-stddev, len(y)))

    plt.title("Background events")
    plt.show()

"""
"""

plt.figure()
plt.plot(np.linspace(0,1,len(poisson_events)), poisson_events)
plt.show()
"""


#############################
#ampdist

###generate n times the same signal with the same parameters, and measure the standard deviaiton of the sample

esim = TraceSimulation(
    ampSpec="../data/spe_R11920-RM_ap0.0002.dat",
    timeSpec="../data/bb3_1700v_timing.txt",
    pulseShape="../data/pulse_FlashCam_7dynode_v2a.dat",
    background_rate = 1e10,
    gain=3,
    no_signal_duration = 1e3,
    noise=0.8,
)

evts_br, k_evts = esim.simulateBackground(evts)

# pmt signal
times, pmtSig, uncertainty_pmt = esim.simulatePMTSignal(evts_br, k_evts) #TODO : make uncertainty from the simulatePMTSignal, with ampdist.rvs(). For now sufficient


eleSig, uncertainty_ele = esim.simulateElectronics(pmtSig, uncertainty_pmt, times)

# adc signal
stimes, samples, samples_unpro, uncertainty_sampled = esim.simulateADC(times, eleSig, uncertainty_ele, 1)

plt.figure()

times = times[int(len(times)*0.03):int(len(times)*0.9)]
pmtSig = pmtSig[int(len(pmtSig)*0.03):int(len(pmtSig)*0.9)]


stddev = np.sqrt(esim.ampStddev**2*(esim.lamda+esim.lamda**2)+esim.lamda)
print("Mean amp", esim.ampMean)
mean = esim.lamda
print("ampstdev", esim.ampStddev)


print("THEOREICAL---PMT (amp jitter)")
print("STD", stddev)
print("MEAN", mean)
print("SIZE", len(times))

real_std = np.std(pmtSig)
mean = np.mean(pmtSig)


plt.plot(times,pmtSig)
plt.plot(times, np.repeat(mean, len(pmtSig)))

plt.plot(times, np.repeat(mean+real_std, len(pmtSig)))
plt.plot(times, np.repeat(mean-real_std, len(pmtSig)))
plt.title("PMT amp jitter")
plt.show()


##############
#electronics


print("THEOREICAL---Electronics")
print("pulseshape area", esim.singePE_area)
print("STD", np.sqrt(stddev**2*esim.pulseShape_TotalSomation))
print("Mean", esim.singePE_area*esim.background_rate*1e-9)

eleSig = eleSig[int(len(eleSig)*0.03):int(len(eleSig)*0.9)]

real_std = np.std(eleSig)
mean = np.mean(eleSig)

plt.plot(times,eleSig)
plt.plot(times, np.repeat(mean, len(eleSig)))

plt.plot(times, np.repeat(mean+real_std, len(eleSig)))
plt.plot(times, np.repeat(mean-real_std, len(eleSig)))
plt.title("Elec - convolution")
plt.show()

stddev =  stddev*np.sqrt(esim.pulseShape_TotalSomation)

##############ADC
print("THEORECIAL --- ADC")
standard_dev = np.sqrt(esim.ampStddev**2*(esim.lamda+esim.lamda**2)+esim.lamda)*np.sqrt(esim.pulseShape_TotalSomation)*esim.gain+esim.noise
print("STD", standard_dev)

real_std = np.std(samples)
mean = np.mean(samples)

plt.figure()
plt.plot(stimes, samples)
plt.plot(stimes, np.repeat(mean, len(samples)))
plt.plot(stimes, np.repeat(mean+real_std, len(samples)))
plt.plot(stimes, np.repeat(mean-real_std, len(samples)))
plt.show()



#########################
#let's plot that
#####
#increase in standard deviaion (so minus the standard deviation of the poisson events) vs the standard deviaiton of the ampspec

"""
variations = []
std = []

for i in np.linspace(0.001, 1, num=20):
	esim = TraceSimulation(
	    #ampSpec="data/spe_R11920-RM_ap0.0002.dat",
	    timeSpec="data/bb3_1700v_timing.txt",
	    pulseShape="data/pulse_FlashCam_7dynode_v2a.dat",
	    background_rate = 1e10,
	    gain=1,
	    no_signal_duration = 1e4,
	    noise=0,
	    variation_of_ampdist_std = i,
	)

	evts_br, k_evts = esim.simulateBackground(evts)

	# pmt signal
	times, pmtSig, uncertainty_pmt = esim.simulatePMTSignal(evts_br, k_evts) #TODO : make uncertainty from the simulatePMTSignal, with ampdist.rvs(). For now sufficient


	eleSig, uncertainty_ele = esim.simulateElectronics(pmtSig, uncertainty_pmt, times)

	# adc signal
	stimes, samples, samples_unpro, uncertainty_sampled = esim.simulateADC(times, eleSig, uncertainty_ele, 1)

	variations.append(i)
	std.append(np.std(pmtSig))

plt.figure()
plt.plot(variations, std)
#plt.plot(variations, [x**2 for x in variations])
plt.show()
"""
