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

class Under_c:


	def execute(esim_init):

		pulse = Pulser(step=esim_init.t_step, pulse_type="none")

		#This will obv generate no events
		evts = pulse.generate_all()
		plt.figure()
		plt.title("bl underestimate")

		means_ = []
		brate = []

		for j in np.logspace(3.0, 9.0, num=5):

		    mean_ = []
		    brate.append(math.log10(j))

		    for i in range(2):
		        #Repeat and generate n different signals with same uncertainties
		        print(i,j)

		        esim = TraceSimulation(
		            ampSpec="data/spe_R11920-RM_ap0.0002.dat",
		            timeSpec="data/bb3_1700v_timing.txt",
		            pulseShape="data/pulse_FlashCam_7dynode_v2a.dat",
		            background_rate = j,
		        )

		        #we need to add random evts that follow a negative exponential for the background rate

		        evts_br, k_evts = esim.simulateBackground(evts)

		        # pmt signal
		        times, pmtSig, uncertainty_pmt = esim.simulatePMTSignal(evts_br, k_evts) #TODO : make uncertainty from the simulatePMTSignal, with ampdist.rvs(). For now sufficient


		        eleSig, uncertainty_ele = esim.simulateElectronics(pmtSig, uncertainty_pmt, times)

		        # adc signal
		        stimes, samples, samples_unpro, uncertainty_sampled = esim.simulateADC(times, eleSig, uncertainty_ele)

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
		        
		        th_bg = np.ones(len(stimes))*(esim.singePE_area*esim.gain*esim.background_rate* 1e-9 + esim.offset)

		        mean_.append(bl_mean)

		    means_.append(esim.singePE_area*esim.gain*esim.background_rate* 1e-9 + esim.offset - statistics.fmean(mean_))


		z = np.polyfit(brate, means_, 1)
		slope_ = z[0]
		offset_ = z[1]
		print(slope_, offset_)

		plt.plot(brate, [x * slope_ + offset_ for x in brate])
		plt.plot(brate, means_)
		plt.show()

