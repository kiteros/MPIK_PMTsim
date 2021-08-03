#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm, rv_histogram

# sampling rate 1/ns
f_sample = 250
# sampling time ns
t_sample = 1/f_sample
# oversampling factor
oversamp = 10
# time step of signal ns
t_step = t_sample/oversamp
# extension time ns (time before first/after last event)
t_ext = 5
# ADC offset LSB
offset = 10
# noise std. dev. LSB
noise = 0.8
# ADC gain LSB
gain = 15


# times of photo electrons in ns
def getPETimes():
	return simulatePETimes(10)

# time spectrum of photo electrons
def getTimeSpectrum():
	return simulateTimeSpectrum()

# amplitude spectrum of photo electrons
def getAmplitudeSpectrum():
	return simulateAmplitudeSpectrum()

def getPulseShape():
	return simulatePulseShape()

# simulates PE times
# npe - number of photo electrons
# returns time points array ns
def simulatePETimes(npe, t_mu = 0, t_sig = 5):
	return norm.rvs(t_mu,t_sig,npe)

# simulates time spectrum of photo electrons (normal distributed)
def simulateTimeSpectrum(t_mu = 1, t_sig = 1):
	return norm.pdf(np.arange(int((t_mu+3*t_sig)/t_step))*t_step,t_mu,t_sig)

# simulates amplitude spectrum of photo electrons (normal distributed)
def simulateAmplitudeSpectrum(t_mu = 1, t_sig = 1):
	return norm.pdf(np.arange(int((t_mu+3*t_sig)/t_step))*t_step,t_mu,t_sig)
	
# simulates pulse shape (normal distributed)
def simulatePulseShape(t_mu = 1, t_sig = 0.3):
	return norm.pdf(np.arange(int((t_mu+3*t_sig)/t_step))*t_step,t_mu,t_sig)

# simulates PMT response to PEs
# returns times array ns
#	signal array au
def simulatePMTSignal(peTimes,timeSpec,ampSpec):
	# make discrete times
	t_min = peTimes.min()-t_ext
	t_max = peTimes.max()+t_ext
	times = np.arange((int(t_max-t_min)/t_step))*t_step+t_min
	# get distributions of spectra
	#TODO avoid errors/optimize?
	timeDist = rv_histogram((timeSpec,np.arange(timeSpec.shape[0]+1)*t_step))
	ampDist = rv_histogram((ampSpec,np.arange(ampSpec.shape[0]+1)*t_step))
	# make signal
	signal = np.zeros(times.shape)
	for t in peTimes:
		t += timeDist.rvs()
		signal[int((t-t_min)/t_step)] += ampDist.rvs() #TODO correct?
	return times, signal


# --- start ---
plt.ion()
# get PE times
peTimes = getPETimes()
plt.scatter(peTimes,np.zeros(peTimes.shape))
# simulate pmt
times, signal = simulatePMTSignal(peTimes, getTimeSpectrum(),getAmplitudeSpectrum())
plt.plot(times, signal)
# convolve with pulse shape
signal = np.convolve(signal, getPulseShape(), 'same') #TODO optimize?
plt.plot(times, signal)
plt.figure()

# make samples
stimes = times[::oversamp]
samples = signal[::oversamp]*gain + norm.rvs(offset,noise,stimes.shape)
samples = samples.astype(int)
plt.plot(stimes, samples)









