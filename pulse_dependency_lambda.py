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
import csv
import scipy.fftpack

from scipy import special


def get_enbw(times, signal):
	L = times[-1]
	t0=-L
	x = np.linspace(t0, -t0, num=1000)
	Len = len(x) # lenght buffer



	dt=x[1]-x[0]
	#Define function
	f=signal

	g=fft(f)
	w = np.fft.fftfreq(f.size)*2*np.pi/dt

	g*=dt*np.exp(-complex(0,1)*w*t0)/(np.sqrt(2*np.pi))

	g = np.abs(g)
	g = g[0:Len//2]
	w = w[0:Len//2]

	df = w[2]-w[1]
	enbw_ = 0
	for i in range(len(w)):
		enbw_+= df*g[i]**2
	enbw_ = enbw_/(g[0]**2)

	return enbw_

def get_theoretical_enbw(lamda, sigma):
	scaling_y = 0.42
	th_enbw = (lamda/4)*(np.exp(sigma**2*lamda**2))*special.erfc(sigma*lamda)/(scaling_y**2)
	return th_enbw

esim = TraceSimulation(
    ampSpec="data/spe_R11920-RM_ap0.0002.dat",
    timeSpec="data/bb3_1700v_timing.txt",
    #pulseShape="data/pulse_FlashCam_7dynode_v2a.dat",
    background_rate = 1e3,
    gain=10,
    no_signal_duration = 1e6,

    ps_mu = 15.11,
    ps_amp = 22.0,
    ps_lambda = 0.0659,
    ps_sigma = 2.7118,
)

pulse = Pulser(step=esim.t_step, pulse_type="none")
evts = pulse.generate_all()

sigmas = []
lamdas = []
coeff = []
enbw = []
enbwratio = []
amplitudes = []
averages = []

th_enbw = []

fig, axs = plt.subplots(2)

for k in np.linspace(22.0, 22.0, num=1):

	for i in np.linspace(0.1, 10, num=10):

		for j in np.linspace(0.001, 0.2, num=40):



			####get the offset from a super low brate

			esim_offset = TraceSimulation(
			    ampSpec="data/spe_R11920-RM_ap0.0002.dat",
			    timeSpec="data/bb3_1700v_timing.txt",
			    #pulseShape="data/pulse_FlashCam_7dynode_v2a.dat",
			    background_rate = 1e3,
			    gain=10,
			    no_signal_duration = 1e4,

			    ps_mu = 15.11,
		        ps_amp = k,
		        ps_lambda = j,
		        ps_sigma = i,
			)

			evts_br, k_evts = esim_offset.simulateBackground(evts)
			times, pmtSig, uncertainty_pmt = esim_offset.simulatePMTSignal(evts_br, k_evts)
			eleSig, uncertainty_ele = esim_offset.simulateElectronics(pmtSig, uncertainty_pmt, times)
			stimes, samples, samples_unpro, uncertainty_sampled = esim_offset.simulateADC(times, eleSig, uncertainty_ele, 1)
			offset, _, _, _, _, _, _, _, _, _ = esim_offset.FPGA(stimes, samples, samples_unpro, uncertainty_sampled, 1, True)

			esim_init = TraceSimulation(
			    ampSpec="data/spe_R11920-RM_ap0.0002.dat",
			    timeSpec="data/bb3_1700v_timing.txt",
			    #pulseShape="data/pulse_FlashCam_7dynode_v2a.dat",
			    background_rate = 3e9,
			    gain=10,
			    no_signal_duration = 1e4,

			    ps_mu = 15.11,
		        ps_amp = k,
		        ps_lambda = j,
		        ps_sigma = i,
			)

			

			evts_br, k_evts = esim_init.simulateBackground(evts)
			times, pmtSig, uncertainty_pmt = esim_init.simulatePMTSignal(evts_br, k_evts)
			eleSig, uncertainty_ele = esim_init.simulateElectronics(pmtSig, uncertainty_pmt, times)
			stimes, samples, samples_unpro, uncertainty_sampled = esim_init.simulateADC(times, eleSig, uncertainty_ele, 1)
			bl_mean, s_mean, std, std_unpro, bl_mean_uncertainty, bl_array, stddev_uncert_mean, stddev_mean, spike, skew = esim_init.FPGA(stimes, samples, samples_unpro, uncertainty_sampled, 1, True)
			ratio = (stddev_mean**2)/(bl_mean-offset)
			ratio = ratio/10


			
			lamdas.append(j)

			#enbw.append(get_enbw(esim_init.pulseShape[0], esim_init.pulseShape[1]))
			th_enbw.append(get_theoretical_enbw(j,i))
			averages.append(get_theoretical_enbw(j,i)/ratio)
			coeff.append(ratio)

		axs[1].plot(th_enbw, coeff, label="Numerical ENBW")
		axs[0].plot(lamdas, averages, label="Theoretical ENBW")
		enbw = []
		coeff = []
		th_enbw = []
		lamdas = []
		averages = []


		sigmas.append(i)
		
axs[1].set_xlabel("ENBW [GHz]")
axs[1].set_ylabel("eta")
axs[1].legend(loc="upper right")
plt.show()
