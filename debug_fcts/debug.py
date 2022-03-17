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


        #Vary everything with respect to everything


        stddev_signal = []
        stddev_baseline = []
        signal_mean = []
        baseline_mean = []
        noise = []
        for i in np.linspace(0.0, 150, num=7):
            #Repeat and generate n different signals with same uncertainties

            esim_init = TraceSimulation(
                ampSpec="data/spe_R11920-RM_ap0.0002.dat",
                timeSpec="data/bb3_1700v_timing.txt",
                pulseShape="data/pulse_FlashCam_7dynode_v2a.dat",
                noise=i,
            )

            #we need to add random evts that follow a negative exponential for the background rate

            evts_br, k_evts = esim_init.simulateBackground(evts)

            # pmt signal
            times, pmtSig, uncertainty_pmt = esim_init.simulatePMTSignal(evts_br, k_evts) #TODO : make uncertainty from the simulatePMTSignal, with ampdist.rvs(). For now sufficient


            eleSig, uncertainty_ele = esim_init.simulateElectronics(pmtSig, uncertainty_pmt, times)

            # adc signal
            stimes, samples, samples_unpro, uncertainty_sampled = esim_init.simulateADC(times, eleSig, uncertainty_ele,1)

            bl_mean, s_mean, std, std_unpro, bl_mean_uncertainty, bl_array,stddev_uncert_mean, stddev_mean = esim_init.FPGA(stimes, samples, samples_unpro, uncertainty_sampled)

            th_bg = np.ones(len(stimes))*(esim_init.singePE_area*esim_init.gain*esim_init.background_rate* 1e-9 + esim_init.offset)
                
            stddev_signal.append(stddev_mean)
            stddev_baseline.append(std)
            signal_mean.append(s_mean)
            baseline_mean.append(bl_mean)
            noise.append(i)

        plt.figure()

        plt.plot(noise, stddev_baseline, label="baseline")
        plt.plot(noise, stddev_signal, label="signal")
        plt.plot(noise, signal_mean, label="signal_mean")
        plt.plot(noise, baseline_mean, label="baseline_mean")

        plt.xlabel("noise")
        plt.ylabel("stddev")
        plt.legend(loc="upper left")

        #################################
        stddev_signal = []
        stddev_baseline = []
        signal_mean = []
        baseline_mean = []
        bg = []
        for i in np.linspace(10**2.0, 10**9, num=5):
            #Repeat and generate n different signals with same uncertainties

            esim_init = TraceSimulation(
                ampSpec="data/spe_R11920-RM_ap0.0002.dat",
                timeSpec="data/bb3_1700v_timing.txt",
                pulseShape="data/pulse_FlashCam_7dynode_v2a.dat",
                background_rate=i,
            )

            #we need to add random evts that follow a negative exponential for the background rate

            evts_br, k_evts = esim_init.simulateBackground(evts)

            # pmt signal
            times, pmtSig, uncertainty_pmt = esim_init.simulatePMTSignal(evts_br, k_evts) #TODO : make uncertainty from the simulatePMTSignal, with ampdist.rvs(). For now sufficient


            eleSig, uncertainty_ele = esim_init.simulateElectronics(pmtSig, uncertainty_pmt, times)

            # adc signal
            stimes, samples, samples_unpro, uncertainty_sampled = esim_init.simulateADC(times, eleSig, uncertainty_ele, 1)

            bl_mean, s_mean, std, std_unpro, bl_mean_uncertainty, bl_array,stddev_uncert_mean, stddev_mean = esim_init.FPGA(stimes, samples, samples_unpro, uncertainty_sampled)

            th_bg = np.ones(len(stimes))*(esim_init.singePE_area*esim_init.gain*esim_init.background_rate* 1e-9 + esim_init.offset)
                
            stddev_signal.append(stddev_mean)
            stddev_baseline.append(std)

            signal_mean.append(s_mean)
            baseline_mean.append(bl_mean)
            bg.append(i)

        plt.figure()

        plt.plot(bg, stddev_baseline, label="baseline")
        plt.plot(bg, stddev_signal, label="signal")

        plt.plot(bg, signal_mean, label="signal_mean")
        plt.plot(bg, baseline_mean, label="baseline_mean")

        plt.xlabel("bacgrkound rate")
        plt.ylabel("stddev")
        plt.legend(loc="upper left")



        #################################
        stddev_signal = []
        stddev_baseline = []
        signal_mean = []
        baseline_mean = []
        gain = []
        for i in np.linspace(2, 15, num=5):
            #Repeat and generate n different signals with same uncertainties

            esim_init = TraceSimulation(
                ampSpec="data/spe_R11920-RM_ap0.0002.dat",
                timeSpec="data/bb3_1700v_timing.txt",
                pulseShape="data/pulse_FlashCam_7dynode_v2a.dat",
                gain=i,
            )

            #we need to add random evts that follow a negative exponential for the background rate

            evts_br, k_evts = esim_init.simulateBackground(evts)

            # pmt signal
            times, pmtSig, uncertainty_pmt = esim_init.simulatePMTSignal(evts_br, k_evts) #TODO : make uncertainty from the simulatePMTSignal, with ampdist.rvs(). For now sufficient


            eleSig, uncertainty_ele = esim_init.simulateElectronics(pmtSig, uncertainty_pmt, times)

            # adc signal
            stimes, samples, samples_unpro, uncertainty_sampled = esim_init.simulateADC(times, eleSig, uncertainty_ele, 1)

            bl_mean, s_mean, std, std_unpro, bl_mean_uncertainty, bl_array,stddev_uncert_mean, stddev_mean = esim_init.FPGA(stimes, samples, samples_unpro, uncertainty_sampled)

            th_bg = np.ones(len(stimes))*(esim_init.singePE_area*esim_init.gain*esim_init.background_rate* 1e-9 + esim_init.offset)
                
            stddev_signal.append(stddev_mean)
            stddev_baseline.append(std)

            signal_mean.append(s_mean)
            baseline_mean.append(bl_mean)
            gain.append(i)

        plt.figure()

        plt.plot(gain, stddev_baseline,label="baseline")
        plt.plot(gain, stddev_signal, label="signal")

        
        plt.plot(gain, signal_mean, label="signal_mean")
        plt.plot(gain, baseline_mean, label="baseline_mean")

        plt.xlabel("gain")
        plt.ylabel("stddev")
        plt.legend(loc="upper left")

        plt.show()