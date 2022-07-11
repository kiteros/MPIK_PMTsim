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

# test function
def fit_function(data, a, b, c, d, e, f, g, h):
    x = data[0]
    y = data[1]
    return a * (x**b) * (y**c) + d*x**e + f*y**g + h

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

def integrateSignal(times, signal):
    """
    Integrates the input signal

    Parameters
    ----------
    times - float
            domain of times
    signal - float
            signal arraz

    Returns
    -------
    sum_
            Integration of the signal
    """
    t_step = times[1]-times[0]

    sum_ = 0
    for i in signal:
        sum_ += i*t_step # maybe wrong
    return sum_


####fourier transform

T = 1/esim.pulseShape[0][-1]
N = len(esim.pulseShape[0])

yf = scipy.fftpack.fft(esim.pulseShape[1])
xf = np.linspace(0.0, 1.0//(2.0*T), N//2)

yf = 2.0/N * np.abs(yf[:N//2])


#### integrate the transform
inte = integrateSignal(xf, yf)
max_ = max(yf)
print("integrale", inte/max_)



pulse = Pulser(step=esim.t_step, pulse_type="none")
evts = pulse.generate_all()

sigmas = []
lamdas = []
coeff = []
enbw = []
enbwratio = []
amplitudes = []
averages = []
eta = []
eta_th = []



offset_E = 0.23
offset_eta = -0.8

first_lamda = True
counter=0

for l in np.linspace(0.03, 0.8, num=2):

	plt.figure()

	for k in np.linspace(1, 40, num=2):

		for i in np.linspace(1, 5, num=3):

			####get the offset from a super low brate

			esim_offset = TraceSimulation(
			    ampSpec="../data/spe_R11920-RM_ap0.0002.dat",
			    timeSpec="../data/bb3_1700v_timing.txt",
			    #pulseShape="data/pulse_FlashCam_7dynode_v2a.dat",
			    background_rate = 1e3,
			    gain=10,
			    no_signal_duration = 1e4,

			    ps_mu = 15.11,
		        ps_amp = k,
		        ps_lambda = l,
		        ps_sigma = i,
			)

			evts_br, k_evts = esim_offset.simulateBackground(evts)
			times, pmtSig, uncertainty_pmt = esim_offset.simulatePMTSignal(evts_br, k_evts)
			eleSig, uncertainty_ele = esim_offset.simulateElectronics(pmtSig, uncertainty_pmt, times)
			stimes, samples, samples_unpro, uncertainty_sampled = esim_offset.simulateADC(times, eleSig, uncertainty_ele, 1)
			offset, _, _, _, _, _, _, _, _, _ = esim_offset.FPGA(stimes, samples, samples_unpro, uncertainty_sampled, 1, True)

			esim_init = TraceSimulation(
			    ampSpec="../data/spe_R11920-RM_ap0.0002.dat",
			    timeSpec="../data/bb3_1700v_timing.txt",
			    #pulseShape="data/pulse_FlashCam_7dynode_v2a.dat",
			    background_rate = 3e9,
			    gain=10,
			    no_signal_duration = 1e4,

			    ps_mu = 15.11,
		        ps_amp = k,
		        ps_lambda = l,
		        ps_sigma = i,
			)

			evts_br, k_evts = esim_init.simulateBackground(evts)
			times, pmtSig, uncertainty_pmt = esim_init.simulatePMTSignal(evts_br, k_evts)
			eleSig, uncertainty_ele = esim_init.simulateElectronics(pmtSig, uncertainty_pmt, times)
			stimes, samples, samples_unpro, uncertainty_sampled = esim_init.simulateADC(times, eleSig, uncertainty_ele, 1)
			bl_mean, s_mean, std, std_unpro, bl_mean_uncertainty, bl_array, stddev_uncert_mean, stddev_mean, spike, skew = esim_init.FPGA(stimes, samples, samples_unpro, uncertainty_sampled, 1, True)
			ratio = (stddev_mean**2)/(bl_mean-offset)
			ratio = ratio/10 #divide by true gain

			####fourier transform

			T = 1/esim_init.pulseShape[0][-1]
			N = len(esim_init.pulseShape[0])

			yf = scipy.fftpack.fft(esim_init.pulseShape[1])
			xf = np.linspace(0.0, 1.0//(2.0*T), N//2)
			yf = 2.0/N * np.abs(yf[:N//2])

			#### integrate the transform
			inte = integrateSignal(xf, yf)
			max_ = max(yf)
			enbw_ = inte/max_

			###fetch from lambda = 0.02
			#########################################################################################
			esim_init_lamda = TraceSimulation(
			    ampSpec="../data/spe_R11920-RM_ap0.0002.dat",
			    timeSpec="../data/bb3_1700v_timing.txt",
			    #pulseShape="data/pulse_FlashCam_7dynode_v2a.dat",
			    background_rate = 3e9,
			    gain=10,
			    no_signal_duration = 1e4,

			    ps_mu = 15.11,
		        ps_amp = k,
		        ps_lambda = 0.02,
		        ps_sigma = i,
			)

			evts_br, k_evts = esim_init_lamda.simulateBackground(evts)
			times, pmtSig, uncertainty_pmt = esim_init_lamda.simulatePMTSignal(evts_br, k_evts)
			eleSig, uncertainty_ele = esim_init_lamda.simulateElectronics(pmtSig, uncertainty_pmt, times)
			stimes, samples, samples_unpro, uncertainty_sampled = esim_init_lamda.simulateADC(times, eleSig, uncertainty_ele, 1)
			bl_mean, s_mean, std, std_unpro, bl_mean_uncertainty, bl_array, stddev_uncert_mean, stddev_mean, spike, skew = esim_init_lamda.FPGA(stimes, samples, samples_unpro, uncertainty_sampled, 1, True)
			ratio_lamda = (stddev_mean**2)/(bl_mean-offset)
			offset_eta = ratio_lamda/10 #divide by true gain

			####fourier transform

			T = 1/esim_init_lamda.pulseShape[0][-1]
			N = len(esim_init_lamda.pulseShape[0])

			yf = scipy.fftpack.fft(esim_init_lamda.pulseShape[1])
			xf = np.linspace(0.0, 1.0//(2.0*T), N//2)
			yf = 2.0/N * np.abs(yf[:N//2])

			#### integrate the transform
			inte = integrateSignal(xf, yf)
			max_ = max(yf)
			offset_ENBW = inte/max_
			#########################################################33

			


			###
			eta_th_ = (enbw_-offset_ENBW)/(-(k**(-1.01)*10**(-0.22))*i+(k**(-0.98))*(10**0.69))+offset_eta

			print("th", eta_th_)
			print("ratio", ratio)

			eta_th.append(eta_th_)
			eta.append(ratio)
			sigmas.append(i)

		amplitudes.append(k)

		plt.plot(sigmas, eta, label="A="+str(k))
		plt.plot(sigmas, eta_th, '--', label="A="+str(k))
		sigmas = []
		eta = []
		eta_th = []

	plt.xlabel("sigmas")
	plt.ylabel("eta")
	plt.legend(loc="upper left")
	plt.savefig('images/'+str(counter)+str(l)+'.png')
	counter += 1
plt.show()
