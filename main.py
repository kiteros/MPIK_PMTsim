#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm, rv_histogram, randint
from scipy.signal import resample
import uproot

class ElectronicsSimulation:
	"""
	Class to simulate PMT and ADC response to photo electron events.

	Use `simulateAll()` to simulate full electronics chain or methods for
	individual simulations if intermiediate results are required.

	Attributes
	----------
	f_sample : float
		sampling rate of ADC in 1/ns
	t_sample : float
		sampling time of ADC in ns
	oversamp : int
		oversampling factor
	t_step : float
		time step for simulation in ns (`t_step=t_sample/oversamp`)
	t_pad : float
		padding before first and after last event in ns
	offset : int
		ADC offset in LSB
	gain : int
		ADC gain (scale factor of normalized amplitude) in LSB
	noise : float
		standard deviation of gaussian noise in LSB
	jitter : int or None
		offset of ADC samples relative to simulation times, `None` for random jitter
	timeSpec : tuple of array_like
		PE delay times in ns and their probabilities
	timeDist : rv_histogram
		distribution of PE delay times
	ampSpec : tuple of array_like
		PE amplitudes in normalized units and their probabilities
	ampsDist : rv_histogram
		distribution of PE amplitudes
	pulseShape : tuple of array_like
		times in ns and amplitudes in normalized units of pulse shape
	plotOffset : float
		time offset for nice plotting in ns
	debugPlots : bool
		plot intermediate results if `True`
	"""
	
	def __init__(self, f_sample = 0.25, oversamp = 10, t_pad = 200, offset = 10, noise = 0.8,
			gain = 15, jitter = None, debugPlots = False,
			timeSpec = None, ampSpec = None, pulseShape = None):
		"""
		Initializes instances of ElectronicsSimulation class.

		`timeSpec`, `ampSpec` and `pulseShape` can have special values:
			'None' -- simulates gaussian spectra/pulse shape
			string -- loads spectra/pulse shape from file with given name
		"""
		self.f_sample = f_sample
		self.t_sample = 1/f_sample
		self.oversamp = oversamp
		self.t_step = self.t_sample/oversamp
		self.t_pad = t_pad
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
				hist, bins = np.histogram(self.timeSpec, binCnt, density=True)
				self.timeSpec = (bins[:-1], hist)
				self.timeDist = rv_histogram((hist,bins))
		else:
			self.timeSpec = timeSpec
		if ampSpec == None:
			self.ampSpec = self.simulateAmplitudeSpectrum()
			#TODO histogram?
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
		sx = self.timeSpec[0]
		bins = np.append(sx, sx[-1]+sx[1]-sx[0])
		self.timeDist = rv_histogram((self.timeSpec[1],bins))
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
	

	def simulatePETimes(self, npe = 10, t_mu = 0, t_sig = 300):
		"""
		Simulates gaussian photo electron arrival times.

		Parameters
		----------
		npe - int
			number of photo electrons
		t_mu - float
			mean time in ns
		t_sig - float
			standard deviation in ns
		
		Returns
		-------
		ndarray
			list of times in ns
		"""
		return norm.rvs(t_mu,t_sig,npe)
	

	def simulateTimeSpectrum(self, t_mu = 1, t_sig = 1):
		"""
		Simulates a gaussian time spectrum.

		Parameters
		----------
		t_mu - float
			mean time in ns
		t_sig - float
			standard deviation in ns
		
		Returns
		-------
		tuple of ndarray
			times in ns and their probabilities
		"""
		tsx = np.arange(int((t_mu+3*t_sig)/self.t_step))*self.t_step
		return tsx, norm.pdf(tsx,t_mu,t_sig)
	
	
	def simulateAmplitudeSpectrum(self, a_mu = 1, a_sig = 1):
		"""
		Simulates a gaussian amplitude spectrum.

		Parameters
		----------
		a_mu - float
			mean amplitude (set to 1 for nomalized output)
		a_sig - float
			standard deviation
		
		Returns
		-------
		tuple of ndarray
			amplitudes and their probabilities
		"""
		asx = np.arange(int((a_mu+3*a_sig)/0.01))*0.01
		return asx, norm.pdf(asx,a_mu,a_sig)
		
	
	def simulatePulseShape(self, t_mu = 0, t_sig = 3):
		"""
		Simulates a gaussian pulse shape.

		Parameters
		----------
		t_mu - float
			peak time in ns
		t_sig - float
			pulse width in ns
		
		Returns
		-------
		tuple of ndarray
			times in ns and amplitudes in normalized units
		"""
		psx = np.arange(int((6*t_sig)/self.t_step))*self.t_step-3*t_sig
		return psx, norm.pdf(psx,t_mu,t_sig)
	
	
	def simulatePMTSignal(self, peTimes):
		"""
		Simulates PMT signal based on photo electron times.

		Parameters
		----------
		peTimes - array_like
			list of photo electron arrival times in ns
		
		Returns
		-------
		tuple of ndarray
			times in ns and simulated signal
		"""
		# make discrete times
		t_min = peTimes.min()-self.t_pad
		t_max = peTimes.max()+self.t_pad
		times = np.arange((int(t_max-t_min)/self.t_step))*self.t_step+t_min
		# make signal
		signal = np.zeros(times.shape)
		for t in peTimes:
			t += self.timeDist.rvs()
			signal[int((t-t_min)/self.t_step)] += self.ampDist.rvs() #TODO correct?
		return times, signal
	

	def simulateElectronics(self, signal):
		"""
		Simulates effect of electronics on PMT signal.

		Parameters
		----------
		signal - array_like
			signal from PMT (use result of `simulatePMTSignal()`)
		
		Returns
		-------
		ndarray
			simulated signal
		"""
		return np.convolve(signal, self.pulseShape[1], 'same')
	
	
	def simulateADC(self, times, signal):
		"""
		Simulates ADC out based on electronics signal.

		Parameters
		----------
		times - array_like
			times of signal in ns
		signal - array_like
			amplitudes of signal in normalized units (use result of `simulateElectronics`)
		
		Returns
		-------
		tuple of ndarray
			times in ns and simulated output in LSB
		"""
		jitter = self.jitter
		if jitter == None:
			jitter = randint.rvs(0,self.oversamp) #TODO random size?
		stimes = times[jitter::self.oversamp]
		samples = signal[jitter::self.oversamp]*self.gain + norm.rvs(self.offset,self.noise,stimes.shape)
		samples = samples.astype(int)
		return stimes, samples
	
	def simulateAll(self, peTimes):
		"""
		Simulates ADC output based on photo electron times.

		Parameters
		----------
		peTimes - array_like
			list of photo electron arrival times in ns
		
		Returns
		-------
		tuple of ndarray
			times in ns and simulated ADC output in LSB
		"""
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
		# return
		return stimes, samples
	
	def getPETimes(self, source = None):	
		#TODO remove
		if source == None:
			peTimes = self.simulatePETimes()
		elif type(source) is str:
			peTimes = np.loadtxt(source, unpack=True)
		return peTimes
	
# end of class



# --- start ---
if __name__ == "__main__":
	# init class
	esim = ElectronicsSimulation(ampSpec="data/bb3_1700v_spe.txt", timeSpec="data/bb3_1700v_timing.txt", pulseShape="data/bb3_1700v_pulse_shape.txt")
	# get PE times
	events = uproot.lazy("data/DAT000001.root:XCDF")
	for pmtIds, peTimes in zip(events["HAWCSim.PE.PMTID"], events["HAWCSim.PE.Time"]):
		# plot all
		peTimes = peTimes.to_numpy()
		pmtIds = pmtIds.to_numpy()
		plt.figure(1)
		plt.scatter(peTimes, pmtIds, c="grey")
		# pick a few
		pmts = np.random.choice(pmtIds, 5)
		for pmt in pmts:
			pets = peTimes[pmtIds == pmt]
			plt.figure(1)
			plt.scatter(pets,pmt*np.ones(pets.shape))
			times, samples = esim.simulateAll(pets)
			plt.figure(2)
			plt.scatter(pets,np.zeros(pets.shape))
			plt.plot(times+esim.plotOffset, samples)
		break
	# simulate
	#esim.simulateAll(peTimes)

	plt.show()








