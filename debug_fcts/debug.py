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

class Debug:

    def execute(esim_init):

        pulse = Pulser(step=esim_init.t_step, pulse_type="none")

        #This will obv generate no events
        evts = pulse.generate_all()

        plt.figure()
        plt.title("Simulated ADC output")

        for i in range(5):
            #Repeat and generate n different signals with same uncertainties

            esim_init = TraceSimulation(
                ampSpec="data/spe_R11920-RM_ap0.0002.dat",
                timeSpec="data/bb3_1700v_timing.txt",
                pulseShape="data/pulse_FlashCam_7dynode_v2a.dat",
            )

            #we need to add random evts that follow a negative exponential for the background rate

            evts_br, k_evts = esim_init.simulateBackground(evts)

            # pmt signal
            times, pmtSig, uncertainty_pmt = esim_init.simulatePMTSignal(evts_br, k_evts) #TODO : make uncertainty from the simulatePMTSignal, with ampdist.rvs(). For now sufficient


            eleSig, uncertainty_ele = esim_init.simulateElectronics(pmtSig, uncertainty_pmt, times)

            # adc signal
            stimes, samples, samples_unpro, uncertainty_sampled = esim_init.simulateADC(times, eleSig, uncertainty_ele)

            bl_mean, s_mean, std, std_unpro, bl_mean_uncertainty, bl_array,stddev_uncert_mean, stddev_mean = esim_init.FPGA(stimes, samples, samples_unpro, uncertainty_sampled)

            th_bg = np.ones(len(stimes))*(esim_init.singePE_area*esim_init.gain*esim_init.background_rate* 1e-9 + esim_init.offset)
            
            plt.plot(stimes + esim_init.plotOffset, samples, label=i)
            plt.plot(stimes + esim_init.plotOffset, np.ones(len(stimes))*bl_mean)
            #plt.plot(stimes + esim.plotOffset, bl_array)
            plt.plot(stimes + esim_init.plotOffset, th_bg)
            plt.fill_between(stimes + esim_init.plotOffset, [a - b for a, b in zip(th_bg, uncertainty_sampled)], [a + b for a, b in zip(th_bg, uncertainty_sampled)], alpha=0.2)
            
            
            """
            plt.figure()
            plt.title("Electronic signal")
            #plt.scatter(evts_br, np.zeros(evts_br.shape))
            #plt.bar(times, pmtSig)
            plt.plot(times, eleSig, label=i)
            plt.fill_between(times, [a - b for a, b in zip(eleSig, uncertainty_ele)], [a + b for a, b in zip(eleSig, uncertainty_ele)], alpha=0.2)
            plt.xlabel("Time/ns")
            plt.ylabel("Amplitude")
            plt.legend(loc="upper left")

            plt.figure()
            plt.title("PMT signal")
            #plt.scatter(evts_br, np.zeros(evts_br.shape))
            #plt.bar(times, pmtSig)
            plt.plot(times, pmtSig,label=i)
            plt.fill_between(times, [a - b for a, b in zip(pmtSig, uncertainty_pmt)], [a + b for a, b in zip(pmtSig, uncertainty_pmt)], alpha=0.2)
            plt.xlabel("Time/ns")
            plt.ylabel("Amplitude")
            plt.legend(loc="upper left")
            """
        plt.xlabel("Time/ns")
        plt.ylabel("ADC output/LSB")
        plt.legend(loc="upper left")
        plt.show()
