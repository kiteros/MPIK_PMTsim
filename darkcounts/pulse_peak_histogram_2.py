#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import rv_histogram, randint, poisson, expon, exponnorm
from scipy.signal import resample
import scipy.integrate as integrate

import sys

sys.path.insert(0, '/home/jebach/Documents/flashcam/pmt-trace-simulation-master/PMTtraceSIM_draft/simulation')
sys.path.insert(0, '/home/jebach/Documents/flashcam/pmt-trace-simulation-master/PMTtraceSIM_draft/debug_fcts')
sys.path.insert(0, '/home/jebach/Documents/flashcam/pmt-trace-simulation-master/PMTtraceSIM_draft/baselineshift')

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

from scipy.stats import norm
from sklearn.neighbors import KernelDensity

from scipy import signal

###################################
#Generate a low br trace
#count the peaks of the pulse shapes, put their height in an histogram (amplitude vs frequency) - izi
############################################

class PeakHistogram:

	### peak histogram, you give him noise, background rate, and trace lenght and a gain linspace, and it outputs an array
	#of G'/G of the same size of the gain linspace

	def __init__(
        self,
        noise=0.8,
        gain_linspace=np.linspace(4,16,num=4),
        background_rate=1e6,
        trace_lenght=1e6,
        graphs=True,
        verbose=False,
        #####
        prominence = 8,
        window = 'tukey',
        resampling_rate=4,
        kde_banwidth=2,
    ):

	    self.noise = noise 
	    self.gain_linspace = gain_linspace
	    self.background_rate = background_rate
	    self.trace_lenght = trace_lenght
	    self.graphs = graphs
	    self.verbose = verbose
	    #####
	    self.prominence = prominence,
	    self.window = window
	    self.resampling_rate = resampling_rate
	    self.kde_banwidth = kde_banwidth
	    self.trace_lenght = trace_lenght

	    esim_init = TraceSimulation(
		    ampSpec="../data/spe_R11920-RM_ap0.0002.dat",
		    timeSpec="../data/bb3_1700v_timing.txt",
		    pulseShape="../data/pulse_FlashCam_7dynode_v2a.dat",
		    background_rate = 1e7,
		    gain=3,
		    no_signal_duration = 7e4,
		    noise=0.8,
		)



	    self.mode_ = esim_init.ampSpec[0][np.argmax(esim_init.ampSpec[1])]
	    self.maxi_ = np.max(esim_init.ampSpec[1])


	    if self.graphs:
		    plt.figure()
		    plt.plot(*esim_init.ampSpec)

		    

		    plt.vlines(self.mode_, 0, self.maxi_, color='red',linestyle="--", label="mode")

		    plt.xlabel("Amplitude")
		    plt.ylabel("Frequency")

		    print("mode of ampspectrum", self.mode_)

	    self.pulse = Pulser(step=esim_init.t_step, pulse_type="none")
	    self.evts = self.pulse.generate_all()

	def get_relative_gain_array(self):



		################we will only generate one trace, and make it longer to count more

		gain_linspace = self.gain_linspace
		gains_hist_list = []
		gains_bins_list = []
		gains_color_list = []

		restituted_gains = []

		max_values_array = []
		mode_array = []
		maxi_array = []
		mean_array = []
		median_array = []


		X_plot_array = []
		log_dens_array = []

		square_samples = int(ceil(np.sqrt(len(gain_linspace))))

		if self.graphs:
			if len(gain_linspace) == 1:
				plt.figure()
			else:
				fig, ax_sample = plt.subplots(square_samples, square_samples,constrained_layout=True)
		iter_x = 0
		iter_y = 0

		for gain in gain_linspace:

			####for each gain we want to test different parameters and see which one had G'/G closest to 1



			if iter_x >= square_samples:
				iter_x = 0
				iter_y += 1

			esim = TraceSimulation(
			    ampSpec="../data/spe_R11920-RM_ap0.0002.dat",
			    timeSpec="../data/bb3_1700v_timing.txt",
			    #pulseShape="data/pulse_FlashCam_7dynode_v2a.dat",
			    background_rate = 1e6,
			    gain=gain,
			    no_signal_duration = self.trace_lenght,
			    noise=0.8,

			)



			evts_br, k_evts = esim.simulateBackground(self.evts)

			# pmt signal
			times, pmtSig, uncertainty_pmt = esim.simulatePMTSignal(evts_br, k_evts) #TODO : make uncertainty from the simulatePMTSignal, with ampdist.rvs(). For now sufficient


			eleSig, uncertainty_ele = esim.simulateElectronics(pmtSig, uncertainty_pmt, times)

			# adc signal
			stimes, samples, samples_unpro, uncertainty_sampled = esim.simulateADC(times, eleSig, uncertainty_ele, 1)

			############
			#So the baseline doesn't move too much with such low brates, we will assume that 
			#we can truncate for LSB > 203, and catch all the maximums above that

			samples = signal.resample(samples, self.resampling_rate*len(stimes), window=self.window)


			stimes = np.linspace(np.min(stimes), np.max(stimes),self.resampling_rate*len(stimes), endpoint=False)

			max_value = 0

			current_max_values = []
			max_values = []

			current_max_values_stime = []
			max_values_stimes = []

			###maybe use findpeak from scipy ?

			maxs, _ = signal.find_peaks(samples, prominence=self.prominence) ###promeminence devrait dependre du gain ? Non
			max_values_stimes = stimes[maxs]
			max_values = samples[maxs]

			"""

			for i in range(len(samples)):
				if samples[i] >= 205:
					#we are on a peak
					current_max_values.append(samples[i])
					current_max_values_stime.append(stimes[i])
				elif samples[i] < 205 and samples[i-1] >= 205:

					
					max_values.append(np.max(current_max_values))
					arg = np.argmax(current_max_values)
					max_values_stimes.append(current_max_values_stime[arg])
					current_max_values = []
					current_max_values_stime = []

			"""

			####let's print the maxvalues on the signal here

			if self.graphs:

				if len(gain_linspace) == 1:
					plt.plot(stimes, samples)
					plt.scatter(max_values_stimes, max_values, marker='o', color='red')
					plt.xlabel("Number of samples")
					plt.ylabel("LSB")
				else:

					ax_sample[iter_y, iter_x].plot(stimes, samples)
					ax_sample[iter_y, iter_x].scatter(max_values_stimes, max_values, marker='o', color='red')
					ax_sample[iter_y, iter_x].set_xlabel("Number of samples")
					ax_sample[iter_y, iter_x].set_ylabel("LSB")
			
			if(len(max_values) == 0):
				max_values = np.array([200])


			max_values = max_values - np.ones(len(max_values))*esim.offset

			hist, bins = np.histogram(max_values, density=True, bins=30)
			width = (bins[1] - bins[0])
			center = (bins[:-1] + bins[1:]) / 2

			gains_hist_list.append(hist)
			gains_bins_list.append(bins)
			#plt.figure()
			#plt.bar(center, hist, align='center', width=width)


			# ----------------------------------------------------------------------
			# Plot a 1D density example
			
			max_values = max_values[:, np.newaxis]
			X_plot = np.linspace(np.min(max_values)-40, np.max(max_values), 1000)[:, np.newaxis]
			DX = X_plot[1]-X_plot[0]

			color = np.random.rand(3,)

			gains_color_list.append(color)

			kde = KernelDensity(kernel="gaussian", bandwidth=self.kde_banwidth).fit(max_values)
			log_dens = kde.score_samples(X_plot)
			max_values_array.append(max_values)
			X_plot_array.append(X_plot)
			log_dens_array.append(np.exp(log_dens))

			####find the mode
			mode = X_plot[np.argmax(np.exp(log_dens))]
			maxi = np.max(np.exp(log_dens))

			##calculate the mean
			mean = np.average(np.linspace(np.min(max_values), np.max(max_values), num=len(np.exp(log_dens))), weights = np.exp(log_dens))
			mean_val = np.exp(log_dens)[int(mean//DX)]

			##calculate the median
			median = np.median(np.exp(log_dens))
			median_val = np.exp(log_dens)[int(median//DX)]


			mode_array.append((mode,maxi))
			mean_array.append((mean,mean_val))
			median_array.append((median,median_val))

			if self.verbose:
				print("gain", gain, "mode", mode, "mean",mean,"median", median)
				print('restituted gain', mode/self.mode_, mode/0.9, mode*0.9)

			restituted_gains.append(mode/self.mode_)

			iter_x +=1




		###Make the big figure with KDE

		if self.graphs:

			plt.figure()

			for i in range(len(gain_linspace)):
				X_plot = X_plot_array[i]
				log_dens = log_dens_array[i]
				color = gains_color_list[i]
				max_values = max_values_array[i]
				gain = gain_linspace[i]

				plt.plot(
				    X_plot[:, 0],
				    log_dens,
				    color=color,
				    lw=2,
				    linestyle="-",
				    label="gaussian kernel, gain="+str(gain),
				)
				plt.legend(loc="upper right")
				plt.plot(max_values[:, 0], -0.005 - 0.003 * np.random.random(max_values.shape[0]) - 0.009*gain/20, "+", color=color)

				mode, maxi = mode_array[i]
				mean, mean_val = mean_array[i]
				median, median_val = median_array[i]


				plt.vlines(mode, 0, maxi, color=color,linestyle="--", label="mode")
				plt.vlines(mean, 0, mean_val, color=color,linestyle="dashdot", label="median")
				plt.vlines(median_val, 0, median, color=color,linestyle="dotted", label="median")




		####if only on datapoint

		if self.graphs:

			if len(gain_linspace)==1:
				plt.figure()
				bins = gains_bins_list[0]
				hist = gains_hist_list[0]

				width = (bins[1] - bins[0])
				center = (bins[:-1] + bins[1:]) / 2
				plt.bar(center, hist, align='center', width=width, color=gains_color_list[0])
				plt.title("Gain " + str(format(gain_linspace[0],".2f")), y=1.0, pad=-50)

			else:

				square = int(ceil(np.sqrt(len(gain_linspace))))
				fig, ax = plt.subplots(square, square,constrained_layout=True)

				iter_ = 0

				for i in range(square):
					for j in range(square):

						if iter_ < len(gain_linspace):

							bins = gains_bins_list[iter_]
							hist = gains_hist_list[iter_]

							width = (bins[1] - bins[0])
							center = (bins[:-1] + bins[1:]) / 2
							ax[i,j].bar(center, hist, align='center', width=width, color=gains_color_list[iter_])
							ax[i,j].set_title("Gain " + str(format(gain_linspace[iter_],".2f")), y=1.0, pad=-50)


						iter_+=1




		relative_gains_ratio = []
		for i in range(len(gain_linspace)):
			relative_gains_ratio.append(restituted_gains[i]/gain_linspace[i])


		if self.graphs:

			plt.figure()
			
			plt.plot(gain_linspace,relative_gains_ratio)
			plt.xlabel("index")
			plt.ylabel("G'/G")

			plt.show()

		relative_gains_ratio_bis = []
		for i in relative_gains_ratio:
			relative_gains_ratio_bis.append(i[0])

		return relative_gains_ratio_bis


