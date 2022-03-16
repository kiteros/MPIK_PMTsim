#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm, rv_histogram, randint, poisson, expon, exponnorm
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


class BL_shift:


	def execute(esim_init):

		

		# Characterize the baseline shift

		pulse = Pulser(step=esim_init.t_step, pulse_type='none')

		# This will obv generate no events

		evts = pulse.generate_all()
		plt.figure()
		for j in np.linspace(2, 15, num=3):
			bl_mean_array = []
			br_rate = []
			std_array = []
			for i in np.linspace(10 ** 5, 10 ** 9, num=10):
			    esim = TraceSimulation(ampSpec='data/spe_R11920-RM_ap0.0002.dat',
			                           timeSpec='data/bb3_1700v_timing.txt',
			                           pulseShape='data/pulse_FlashCam_7dynode_v2a.dat'
			                           , background_rate=i
			                           ,gain = j)
			    (evts_br, k_evts) = esim.simulateBackground(evts)

			    # pmt signal

			    (times, pmtSig, uncertainty_pmt) = esim.simulatePMTSignal(evts_br,
			            k_evts)  # TODO : make uncertainty from the simulatePMTSignal, with ampdist.rvs(). For now sufficient

			    (eleSig, uncertainty_ele) = esim.simulateElectronics(pmtSig,
			            uncertainty_pmt, times)

			    # adc signal

			    (stimes, samples, samples_unpro, uncertainty_sampled) = \
			        esim.simulateADC(times, eleSig, uncertainty_ele)

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


			    bl_mean_array.append(math.log10(bl_mean))
			    br_rate.append(i)
			    std_array.append(std**2)
			    print("std", std)

			
			plt.plot(std_array, bl_mean_array, label="j")
		#plt.scatter(br_rate, np.zeros(len(br_rate)))
		plt.show()

		return 1


