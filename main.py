#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm, rv_histogram, randint
from scipy.signal import resample

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
	# time distribution
	timeDist = None
	# amplitude spectrum
	ampSpec = None
	# amplitude distribution
	ampDist = None
	# pulse shape
	pulseShape = None
	# time offset for nice plotting
	plotOffset = 0
	
	def __init__(self, f_sample = 0.25, oversamp = 10, t_ext = 200, offset = 10, noise = 0.8,
			gain = 15, jitter = None, debugPlots = False,
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
		self.debugPlots = debugPlots
		#load spectra from file or simulate
		if timeSpec == None:
			self.timeSpec = self.simulateTimeSpectrum()
		elif type(timeSpec) is str:
			self.timeSpec = np.loadtxt(timeSpec, unpack=True)
			if self.timeSpec.ndim == 1:
				binCnt = int((self.timeSpec.max()-self.timeSpec.min())/self.t_step)
				hist, bins = np.histogram(self.timeSpec, binCnt, normed=True)
				self.timeSpec = (bins[:-1], hist)
				self.timeDist = rv_histogram((hist,bins))
		else:
			self.timeSpec = timeSpec
		if ampSpec == None:
			self.ampSpec = self.simulateAmplitudeSpectrum()
		elif type(ampSpec) is str:
			self.ampSpec = np.loadtxt(ampSpec, unpack=True)
		else:
			self.ampSpec = ampSpec
		# load pulse shape from file or simulate
		if pulseShape == None:
			self.pulseShape = self.simulatePulseShape()
		elif type(pulseShape) is str:
			ps = np.loadtxt(pulseShape, unpack=True)
			# ensure correct sampling
			step = ps[0][1]-ps[0][0]
			ps, t = resample(ps[1], int(step/self.t_step*ps[1].shape[0]), ps[0])
			self.pulseShape = (t,ps)
			# offset from center (for plots only)
			self.plotOffset = (t[-1]+t[0])/2
		else:
			self.pulseShape = pulseShape
		# get distributions of spectra
		if self.timeDist == None:
			sx = self.timeSpec[0]
			bins = np.append(sx, sx[-1]+sx[1]-sx[0])
			self.timeDist = rv_histogram((self.timeSpec[1],bins))
		if self.ampDist == None:
			sx = self.ampSpec[0]
			bins = np.append(sx, sx[-1]+sx[1]-sx[0])
			self.ampDist = rv_histogram((self.ampSpec[1],bins))
			
		# figures for debugging
		if self.debugPlots:
			plt.figure()
			plt.plot(*self.pulseShape)
			plt.title("Pulse shape")
			plt.xlabel("t/ns")
			plt.ylabel("A/au")
			plt.figure()
			plt.subplot(1,2,1)
			plt.plot(*self.timeSpec)
			plt.title("Time spectrum")
			plt.xlabel("t/ns")
			plt.ylabel("Probability")
			plt.subplot(1,2,2)
			plt.plot(*self.ampSpec)
			plt.title("Amplitude spectrum")
			plt.xlabel("A/au")
			plt.ylabel("Probability")
	
	# simulates PE times
	# npe - number of photo electrons
	# returns time points array ns
	def simulatePETimes(self, npe = 10, t_mu = 0, t_sig = 300):
		return norm.rvs(t_mu,t_sig,npe)
	
	# simulates time spectrum of photo electrons (normal distributed)
	def simulateTimeSpectrum(self, t_mu = 1, t_sig = 1):
		tsx = np.arange(int((t_mu+3*t_sig)/self.t_step))*self.t_step
		return tsx, norm.pdf(tsx,t_mu,t_sig)
	
	# simulates amplitude spectrum of photo electrons (normal distributed)
	def simulateAmplitudeSpectrum(self, a_mu = 1, a_sig = 1):
		asx = np.arange(int((a_mu+3*a_sig)/0.01))*0.01
		return asx, norm.pdf(asx,a_mu,a_sig)
		
	# simulates pulse shape (normal distributed)
	def simulatePulseShape(self, t_mu = 0, t_sig = 3):
		psx = np.arange(int((6*t_sig)/self.t_step))*self.t_step-3*t_sig
		return psx, norm.pdf(psx,t_mu,t_sig)
	
	# simulates PMT response to PEs
	# returns times array ns
	#	signal array au
	def simulatePMTSignal(self, peTimes):
		# make discrete times
		t_min = peTimes.min()-self.t_ext
		t_max = peTimes.max()+self.t_ext
		times = np.arange((int(t_max-t_min)/self.t_step))*self.t_step+t_min
		# make signal
		signal = np.zeros(times.shape)
		for t in peTimes:
			t += self.timeDist.rvs()
			signal[int((t-t_min)/self.t_step)] += self.ampDist.rvs() #TODO correct?
		return times, signal
	
	# simulates the elctronics
	# returns signal array au
	def simulateElectronics(self, signal):
		return np.convolve(signal, self.pulseShape[1], 'same') #TODO optimize?
	
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
	
	def simulateAll(self, peTimes):
		# simulate pmt
		times, signal = esim.simulatePMTSignal(peTimes)
		# convolve with pulse shape
		signal = esim.simulateElectronics(signal)
		# make samples
		stimes, samples = esim.simulateADC(times, signal)
		# debug plots
		if self.debugPlots:
			plt.figure()
			plt.scatter(peTimes,np.zeros(peTimes.shape))
			plt.plot(times+esim.plotOffset, signal)
			plt.xlabel("t/ns")
			plt.ylabel("A/au")
			plt.title("PMT signal")
			plt.figure()
			plt.plot(stimes, samples)
			plt.title("ADC output")
			plt.xlabel("t/ns")
			plt.ylabel("A/LSB")
	
	def getPETimes(self, source = None):	
		#TODO load from file
		if source == None:
			peTimes = self.simulatePETimes()
		elif type(source) is str:
			peTimes = np.loadtxt(source, unpack=True)
		return peTimes
	
# end of class



# --- start ---
#plt.ion()
# init class
esim = ElectronicsSimulation(ampSpec="data/bb3_1700v_spe.txt", timeSpec="data/bb3_1700v_timing.txt", pulseShape="data/bb3_1700v_pulse_shape.txt", debugPlots=True)
# get PE times
peTimes = esim.getPETimes()
# simulate
esim.simulateAll(peTimes)

plt.show()








