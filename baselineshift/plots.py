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

predict_brate = 1e8


#Amplitude spectrum obtained from spe_R11920-RM_ap0.0002.dat
ampSpec = np.loadtxt("../data/bb3_1700v_spe.txt", unpack=True)
timeSpec = "../data/bb3_1700v_timing.txt"
pulseShape = np.loadtxt("../data/pulse_FlashCam_7dynode_v2a.dat", unpack=True)

# init class
esim_init = TraceSimulation(
    ampSpec="../data/spe_R11920-RM_ap0.0002.dat",
    timeSpec="../data/bb3_1700v_timing.txt",
    pulseShape="../data/pulse_FlashCam_7dynode_v2a.dat",
    show_graph = False,
)


pulse = Pulser(step=esim_init.t_step, pulse_type="none")
evts = pulse.generate_all()

gc = GainCalculator()
gc.train()




fig, axs = plt.subplots(1)

for i in np.linspace(2, 15, num=10):
	line = []
	brate = []
	unc = []

	for j in np.logspace(6,9, num=20):

		#generate it
		esim = TraceSimulation(
		    ampSpec="../data/spe_R11920-RM_ap0.0002.dat",
		    timeSpec="../data/bb3_1700v_timing.txt",
		    pulseShape="../data/pulse_FlashCam_7dynode_v2a.dat",
		    show_graph = False,
		    background_rate = j,
		    gain=i,
		)

		evts_br, k_evts = esim.simulateBackground(evts)

		# pmt signal
		times, pmtSig, uncertainty_pmt = esim.simulatePMTSignal(evts_br, k_evts) #TODO : make uncertainty from the simulatePMTSignal, with ampdist.rvs(). For now sufficient


		eleSig, uncertainty_ele = esim.simulateElectronics(pmtSig, uncertainty_pmt, times)

		# adc signal
		stimes, samples, samples_unpro, uncertainty_sampled = esim.simulateADC(times, eleSig, uncertainty_ele, 1)

		#This part should be done all the time, even when it is loaded
		bl_mean, _, std, _, _, _, _, stddev_mean, spike, skew = esim.FPGA(stimes, samples, samples_unpro, uncertainty_sampled, 1,  False)

		predicted_gain, uncert = gc.esimate(bl_mean, stddev_mean)

		line.append(100*(predicted_gain-i)/i)
		brate.append(j)
		unc.append(100*(uncert)/i)

		print("real", i, j)
		print("extracted", predicted_gain)

	axs.semilogx(brate, line, label="Gain="+str(format(i,".2f")))
	#axs[1].semilogx(brate, unc, label="Gain="+str(i))

axs.set_xlabel("Bacgrkound rate [Hz]")
axs.set_ylabel("Gain estimate Bias [%]")
axs.legend(loc="upper left")
axs.grid()

axs.title.set_text("Gain estimation with 8 gain sample x 8 backgorund rate sample")

#axs[1].set_xlabel("Bacgrkound rate [Hz]")
#axs[1].set_ylabel("Gain standard deviation [\% of predicted gain]")
#axs[1].legend(loc="upper left")
#axs[1].grid()
plt.show()

