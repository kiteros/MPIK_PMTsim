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
from scipy.stats import norm


class BL_stddev:


	def execute(esim_init):

		bl_stddev = []
		uncertainty_stddev = []
		br_rate = []
		stddev_mean_e = []

		bl_stddev_log = []
		uncertainty_stddev_log = []
		br_rate_log = []
		stddev_mean_e_log = []

		pulse = Pulser(step=esim_init.t_step, pulse_type='none')

		# This will obv generate no events

		evts = pulse.generate_all()

		for j in np.logspace(6, 9, num=7):

		    # br_rate.append(j)

		    br_rate_log.append(math.log10(j))
		    br_rate.append(j)

		    print(j)

		    esim = TraceSimulation(ampSpec='data/spe_R11920-RM_ap0.0002.dat',
		                           timeSpec='data/bb3_1700v_timing.txt',
		                           pulseShape='data/pulse_FlashCam_7dynode_v2a.dat'
		                           , background_rate=j)

		    # we need to add random evts that follow a negative exponential for the background rate

		    (evts_br, k_evts) = esim.simulateBackground(evts)

		    # pmt signal

		    (times, pmtSig, uncertainty_pmt) = esim.simulatePMTSignal(evts_br, k_evts)  
		    # TODO : make uncertainty from the simulatePMTSignal, with ampdist.rvs(). For now sufficient

		    (eleSig, uncertainty_ele) = esim.simulateElectronics(pmtSig, uncertainty_pmt, times)

		    # adc signal

		    (stimes, samples, samples_unpro, uncertainty_sampled) = esim.simulateADC(times, eleSig, uncertainty_ele)

		    (
		        bl_mean,
		        s_mean,
		        std,
		        std_unpro,
		        bl_mean_uncertainty,
		        bl_array,
		        stddev_uncert_mean,
		        stddev_mean,
		        ) = esim.FPGA(stimes, samples, samples_unpro,
		                      uncertainty_sampled)

		    th_bg = np.ones(len(stimes)) * (esim.singePE_area * esim.gain
		                                    * esim.background_rate * 1e-9
		                                    + esim.offset)

		    bl_stddev_log.append(math.log10(std))  # baseline
		    bl_stddev.append(std)  # baseline

		    # bl_stddev.append(np.std(bl_array))

		    uncertainty_stddev.append(stddev_uncert_mean)
		    uncertainty_stddev_log.append(math.log10(stddev_uncert_mean))

		    # uncertainty_stddev.append(stddev_uncert_mean)

		    stddev_mean_e.append(stddev_mean)  # signal
		    stddev_mean_e_log.append(math.log10(stddev_mean))  # signal

		z = np.polyfit(br_rate_log, bl_stddev_log, 1)  # final
		z2 = np.polyfit(br_rate_log, stddev_mean_e_log, 1)  # initial

		# self.smoothing_coeff_offset = z[1]-z[0]*z2[1]/z2[0]
		# self.smoothing_coeff_slope = z[0]/z2[0]

		# self.smoothing_coeff_offset = 7.25*z[0]+z[1]-(7.25*z2[0]+z2[1]) ### assuming this value is constant

		smoothing_coeff_offset = np.float32(-0.14140167346889987)

		log_fit_bl = [x * z[0] + z[1] for x in br_rate_log]
		log_fit_signal = [x * z2[0] + z2[1] for x in br_rate_log]

		##transform back to real space, les fits et le tranformed signal
		# log(bl)=z[0]*log(br)+z[1]
		# (log(br)-z[1])=
		# bl = br^(z[0])*10^(z[1])

		transformed_signal = stddev_mean_e_log + smoothing_coeff_offset
		transformed_signal = [10 ** x for x in transformed_signal]

		real_fit_bl = [x ** z[0] * 10 ** z[1] for x in br_rate]
		real_fit_signal = [x ** z2[0] * 10 ** z2[1] for x in br_rate]

		plt.figure()

		"""
		plt.plot(br_rate_log, log_fit_bl, label="bl fit")
		plt.plot(br_rate_log, log_fit_signal, label="stddev_mean fit")

		plt.plot(br_rate_log, uncertainty_stddev_log, label='uncert')
		plt.plot(br_rate_log, bl_stddev_log, label='bl')
		plt.plot(br_rate_log, stddev_mean_e_log, label='stddev mean')
		"""
		
		#plt.plot(br_rate_log, transformed_signal, label='transformed signal')

		
		plt.plot(br_rate, uncertainty_stddev, label='uncert')
		plt.plot(br_rate, bl_stddev, label='bl')
		plt.plot(br_rate, stddev_mean_e, label='stddev mean')
		plt.plot(br_rate, transformed_signal, label='transformed signal')
		
		
		plt.legend(loc='upper left')
		plt.show()

		return 1


