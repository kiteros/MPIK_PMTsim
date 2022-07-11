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

gains = np.linspace(2.0, 50.0, num=6)
ratio = []

all_gains = []

for times in np.linspace(1, 2, num=3):
	#repeat n times

	for gain in gains:

		esim_offset = TraceSimulation(
		    ampSpec="../data/spe_R11920-RM_ap0.0002.dat",
		    timeSpec="../data/bb3_1700v_timing.txt",
		    #pulseShape="data/pulse_FlashCam_7dynode_v2a.dat",
		    background_rate = 1e3,
		    gain=gain,
		    no_signal_duration = 5e4,

		    ps_mu = 15.11,
	        ps_lambda = 0.0659,
	        ps_sigma = 2.7118,
		)

		evts_br, k_evts = esim_offset.simulateBackground(evts)
		times, pmtSig, uncertainty_pmt = esim_offset.simulatePMTSignal(evts_br, k_evts)
		eleSig, uncertainty_ele = esim_offset.simulateElectronics(pmtSig, uncertainty_pmt, times)
		stimes, samples, samples_unpro, uncertainty_sampled = esim_offset.simulateADC(times, eleSig, uncertainty_ele, 1)
		offset, _, _, _, _, _, _, _, _, _ = esim_offset.FPGA(stimes, samples, samples_unpro, uncertainty_sampled, 1, False)


		esim = TraceSimulation(
		    ampSpec="../data/spe_R11920-RM_ap0.0002.dat",
		    timeSpec="../data/bb3_1700v_timing.txt",
		    #pulseShape="data/pulse_FlashCam_7dynode_v2a.dat",
		    background_rate = 1e9,
		    gain=gain,
		    no_signal_duration = 1e4,

		    ps_mu = 15.11,
	        ps_lambda = 0.0659,
	        ps_sigma = 2.7118,
		)

		evts_br, k_evts = esim.simulateBackground(evts)

		times, pmtSig, uncertainty_pmt = esim.simulatePMTSignal(evts_br, k_evts)

		eleSig, uncertainty_ele = esim.simulateElectronics(pmtSig, uncertainty_pmt, times)

		stimes, samples, samples_unpro, uncertainty_sampled = esim.simulateADC(times, eleSig, uncertainty_ele, 1)

		bl_mean, s_mean, std, std_unpro, bl_mean_uncertainty, bl_array, stddev_uncert_mean, stddev_mean, spike, skew = esim.FPGA(stimes, samples, samples_unpro, uncertainty_sampled, 1, True)

	
		ratio.append(stddev_mean**2/(bl_mean-offset))

	all_gains.append(ratio)
	ratio = []

plt.figure()
for i in all_gains:

	plt.plot(gains, i)
plt.plot(gains, [x*np.polyfit(gains, all_gains[0], 1)[0]+np.polyfit(gains, all_gains[0], 1)[1] for x in gains])

plt.xlabel("gains")
plt.ylabel("ratio var/bl_shift")

plt.show()