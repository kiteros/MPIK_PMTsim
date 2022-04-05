#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import rv_histogram, randint, poisson, expon, exponnorm
from scipy.signal import resample
import scipy.integrate as integrate
from pulser import Pulser
import scipy
import math 
from trace_simulation import TraceSimulation
from scipy.optimize import curve_fit
from scipy import odr
from pylab import *
import statistics
import os.path

from debug_fcts.bl_shift import BL_shift
from debug_fcts.bl_stddev import BL_stddev
from debug_fcts.under_c import Under_c
from debug_fcts.debug import Debug
#from debug_fcts.baseline import Baseline
from debug_fcts.pulse import Pulse

from scipy.stats import norm

from mpl_toolkits.mplot3d import axes3d, Axes3D
from sklearn import linear_model 

from calculate_gains import GainCalculator
import pandas as pd

#########
#File is made to study the shift of any gain point between 3e9 and 1e10 Brate dependency with trace lenght
########

####Generatea trace

ratios = []
ratios2 = []
duration = []

baseline1 = []
baseline2 = []
variance1 = []
variance2 = []

esim_init = TraceSimulation(
    ampSpec="data/spe_R11920-RM_ap0.0002.dat",
    timeSpec="data/bb3_1700v_timing.txt",
    pulseShape="data/pulse_FlashCam_7dynode_v2a.dat",
)

for i in np.logspace(4, 6, num=10):

	pulse = Pulser(step=esim_init.t_step, pulse_type="none")

	evts = pulse.generate_all()

	#generate it

	i = int(i)

	print(i)


	esim = TraceSimulation(
	    ampSpec="data/spe_R11920-RM_ap0.0002.dat",
	    timeSpec="data/bb3_1700v_timing.txt",
	    pulseShape="data/pulse_FlashCam_7dynode_v2a.dat",
	    background_rate = 3e9,
	    gain=10,
	    no_signal_duration = i,
	)

	evts_br, k_evts = esim.simulateBackground(evts)

	# pmt signal
	times, pmtSig, uncertainty_pmt = esim.simulatePMTSignal(evts_br, k_evts) 

	eleSig, uncertainty_ele = esim.simulateElectronics(pmtSig, uncertainty_pmt, times)

	# adc signal
	stimes, samples, samples_unpro, uncertainty_sampled = esim.simulateADC(times, eleSig, uncertainty_ele, 1)

	#This part should be done all the time, even when it is loaded
	bl_mean, s_mean, std, std_unpro, bl_mean_uncertainty, bl_array, stddev_uncert_mean, stddev_mean, spike, skew = esim.FPGA(stimes, samples, samples_unpro, uncertainty_sampled, 1, True)
	ratios.append(stddev_mean**2/(bl_mean-199.3))

	baseline1.append(bl_mean-199.3)
	variance1.append(stddev_mean**2)
	duration.append(i)


for i in np.logspace(4, 6, num=10):

	pulse = Pulser(step=esim_init.t_step, pulse_type="none")

	evts = pulse.generate_all()

	#generate it

	i = int(i)

	print(i)


	esim = TraceSimulation(
	    ampSpec="data/spe_R11920-RM_ap0.0002.dat",
	    timeSpec="data/bb3_1700v_timing.txt",
	    pulseShape="data/pulse_FlashCam_7dynode_v2a.dat",
	    background_rate = 5e8,
	    gain=10,
	    no_signal_duration = i,
	)

	evts_br, k_evts = esim.simulateBackground(evts)

	# pmt signal
	times, pmtSig, uncertainty_pmt = esim.simulatePMTSignal(evts_br, k_evts) 

	eleSig, uncertainty_ele = esim.simulateElectronics(pmtSig, uncertainty_pmt, times)

	# adc signal
	stimes, samples, samples_unpro, uncertainty_sampled = esim.simulateADC(times, eleSig, uncertainty_ele, 1)

	#This part should be done all the time, even when it is loaded
	bl_mean, s_mean, std, std_unpro, bl_mean_uncertainty, bl_array, stddev_uncert_mean, stddev_mean, spike, skew = esim.FPGA(stimes, samples, samples_unpro, uncertainty_sampled, 1, True)
	ratios2.append(stddev_mean**2/(bl_mean-199.3))
	baseline2.append(bl_mean-199.3)
	variance2.append(stddev_mean**2)


plt.figure()
plt.semilogx(duration, ratios, label="gain=10, br=2e9")
plt.semilogx(duration, ratios2, label="gain=10, br=5e8")
plt.xlabel("Duration in ns")
plt.ylabel("Value of ratio")
plt.legend(loc="upper right")
plt.grid()




plt.figure()
plt.semilogx(duration, baseline1, label="gain=10, br=2e9")
plt.semilogx(duration, baseline2, label="gain=10, br=5e8")
plt.xlabel("Duration in ns")
plt.ylabel("Baseline-offset [LSB]")
plt.legend(loc="upper right")
plt.grid()


plt.figure()
plt.semilogx(duration, variance1, label="gain=10, br=2e9")
plt.semilogx(duration, variance2, label="gain=10, br=5e8")
plt.xlabel("Duration in ns")
plt.ylabel("Variance")
plt.legend(loc="upper right")
plt.grid()


plt.show()