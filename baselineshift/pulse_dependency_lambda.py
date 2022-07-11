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


plt.rcParams['text.usetex'] = True


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
	#th_enbw = (lamda/4)*(np.exp(sigma**2*lamda**2))*special.erfc(sigma*lamda)/(scaling_y**2)

	#We are going to do the same, but ignoring the scaling factor
	th_enbw = (lamda/4)*(np.exp(sigma**2*lamda**2))*special.erfc(sigma*lamda)

	return th_enbw

esim = TraceSimulation(
    ampSpec="../data/spe_R11920-RM_ap0.0002.dat",
    timeSpec="../data/bb3_1700v_timing.txt",
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

plt.figure()

coefficient_per_point = []
all_coefficients = []

for k in np.linspace(1.0, 1.0, num=1):

	for i in np.linspace(0.1, 10, num=15):

		for j in np.linspace(0.01, 0.2, num=25):



			####get the offset from a super low brate
			"""
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
			"""
			offset = 199.54005208333334

			esim_init = TraceSimulation(
			    ampSpec="../data/spe_R11920-RM_ap0.0002.dat",
			    timeSpec="../data/bb3_1700v_timing.txt",
			    #pulseShape="data/pulse_FlashCam_7dynode_v2a.dat",
			    background_rate = 3e9,
			    gain=10,
			    no_signal_duration = 1e4,

			    ps_mu = 15.11,
		        ps_amp = k,
		        ps_lambda = j,
		        ps_sigma = i,
		        force_A=True,
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
			averages.append(ratio/get_theoretical_enbw(j,i))
			coeff.append(ratio)

		plt.plot(th_enbw, coeff, label="Numerical ENBW")
		#axs[0].plot(lamdas, averages, label="Theoretical ENBW")
		coefficient_per_point.append(averages[-2])
		enbw = []
		coeff = []
		th_enbw = []
		lamdas = []
		averages = []


		sigmas.append(i)

	C = np.mean(coefficient_per_point)
	all_coefficients.append(C)
	amplitudes.append(k)
	coefficient_per_point = []
		
plt.xlabel("ENBW [GHz]")
plt.ylabel(r'$\eta$')
#plt.legend(loc="upper right")
plt.show()



print("f(A)=", C)

plt.figure()
plt.plot(amplitudes, all_coefficients)
plt.scatter(amplitudes, all_coefficients)
plt.xlabel("A")
plt.ylabel("f(A)")
plt.show()

slope = all_coefficients[-1]/amplitudes[-1]
print("f(a)=", slope, "*A")
