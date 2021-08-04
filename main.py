#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm, rv_histogram, randint

class ElectronicsSimulation:	
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
	# ADC jitter
	jitter = None
	# time spectrum
	timeSpec = None
	# amplitude spectrum
	ampSpec = None
	# pulse shape
	pulseShape = None
	
	def __init__(self, f_sample = 250, oversamp = 10, t_ext = 5, offset = 10, noise = 0.8, gain = 15, jitter = None,
			timeSpec = None, ampSpec = None, pulseShape = None):
		self.f_sample = f_sample
		self.t_sample = 1/f_sample
		self.oversamp = oversamp
		self.t_step = self.t_sample/oversamp
		self.t_ext = t_ext
		self.offset = offset
		self.noise = noise
		self.gain = gain
		self.jitter = jitter
		#TODO load from file
		if timeSpec == None:
			self.timeSpec = self.simulateTimeSpectrum()
		if ampSpec == None:
			self.ampSpec = self.simulateAmplitudeSpectrum()
		if pulseShape == None:
			self.pulseShape = self.simulatePulseShape()
	
	# simulates PE times
	# npe - number of photo electrons
	# returns time points array ns
	def simulatePETimes(self, npe = 10, t_mu = 0, t_sig = 5):
		return norm.rvs(t_mu,t_sig,npe)
	
	# simulates time spectrum of photo electrons (normal distributed)
	def simulateTimeSpectrum(self, t_mu = 1, t_sig = 1):
		return norm.pdf(np.arange(int((t_mu+3*t_sig)/self.t_step))*self.t_step,t_mu,t_sig)
	
	# simulates amplitude spectrum of photo electrons (normal distributed)
	def simulateAmplitudeSpectrum(self, t_mu = 1, t_sig = 1):
		return norm.pdf(np.arange(int((t_mu+3*t_sig)/self.t_step))*self.t_step,t_mu,t_sig)
		
	# simulates pulse shape (normal distributed)
	def simulatePulseShape(self, t_mu = 1, t_sig = 0.3):
		return norm.pdf(np.arange(int((t_mu+3*t_sig)/self.t_step))*self.t_step,t_mu,t_sig)
	
	# simulates PMT response to PEs
	# returns times array ns
	#	signal array au
	def simulatePMTSignal(self, peTimes):
		# make discrete times
		t_min = peTimes.min()-self.t_ext
		t_max = peTimes.max()+self.t_ext
		times = np.arange((int(t_max-t_min)/self.t_step))*self.t_step+t_min
		# get distributions of spectra
		#TODO avoid errors/optimize?/reuse
		timeDist = rv_histogram((self.timeSpec,np.arange(self.timeSpec.shape[0]+1)*self.t_step))
		ampDist = rv_histogram((self.ampSpec,np.arange(self.ampSpec.shape[0]+1)*self.t_step))
		# make signal
		signal = np.zeros(times.shape)
		for t in peTimes:
			t += timeDist.rvs()
			signal[int((t-t_min)/self.t_step)] += ampDist.rvs() #TODO correct?
		return times, signal
	
	# simulates the elctronics
	# returns signal array au
	def simulateElectronics(self, signal):
		return np.convolve(signal, self.pulseShape, 'same') #TODO optimize?
	
	# simulate ADC
	# returns times array ns
	#	samples array LSB
	def simulateADC(self, times, signal):
		jitter = self.jitter
		if jitter == None:
			jitter = randint.rvs(0,self.oversamp) #TODO random size?
		stimes = times[jitter::self.oversamp]
		samples = signal[jitter::self.oversamp]*self.gain + norm.rvs(self.offset,self.noise,stimes.shape)
		samples = samples.astype(int)
		return stimes, samples
	
	def getPETimes(self, source = None):	
		#TODO load from file
		if source == None:
			peTimes = self.simulatePETimes()
		return peTimes
	
# end of class



# --- start ---
plt.ion()
# init class
esim = ElectronicsSimulation()
# get PE times
peTimes = esim.getPETimes()
plt.scatter(peTimes,np.zeros(peTimes.shape))
# simulate pmt
times, signal = esim.simulatePMTSignal(peTimes)
plt.plot(times, signal)
# convolve with pulse shape
signal = esim.simulateElectronics(signal)
plt.plot(times, signal)
plt.figure()

# make samples
stimes, samples = esim.simulateADC(times, signal)
plt.plot(stimes, samples)









