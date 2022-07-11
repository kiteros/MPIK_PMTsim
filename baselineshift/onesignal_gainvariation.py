#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import rv_histogram, randint, poisson, expon, exponnorm
from scipy.signal import resample
import scipy.integrate as integrate

import sys
sys.path.insert(0, '/home/jebach/Documents/flashcam/pmt-trace-simulation-master/PMTtraceSIM_draft/debug_fcts')
sys.path.insert(0, '/home/jebach/Documents/flashcam/pmt-trace-simulation-master/PMTtraceSIM_draft/simulation')
sys.path.insert(0, '/home/jebach/Documents/flashcam/pmt-trace-simulation-master/PMTtraceSIM_draft/darkcounts')

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


# importing pandas package
import pandas as pd

esim_init = TraceSimulation(
    ampSpec="../data/spe_R11920-RM_ap0.0002.dat",
    timeSpec="../data/bb3_1700v_timing.txt",
    pulseShape="../data/pulse_FlashCam_7dynode_v2a.dat",
    background_rate = 3e9,
    gain=10,
    no_signal_duration = 1e4,

)


pulse = Pulser(step=esim_init.t_step, pulse_type="none")
evts = pulse.generate_all()

plt.figure()

for times in range(5):

    esim = TraceSimulation(
        ampSpec="../data/spe_R11920-RM_ap0.0002.dat",
        timeSpec="../data/bb3_1700v_timing.txt",
        pulseShape="../data/pulse_FlashCam_7dynode_v2a.dat",
        background_rate = 3e9,
        gain=2,
        no_signal_duration = 1e5,
        noise=0.8,
    )

    evts_br, k_evts = esim.simulateBackground(evts)

    # pmt signal
    times, pmtSig, uncertainty_pmt = esim.simulatePMTSignal(evts_br, k_evts) #TODO : make uncertainty from the simulatePMTSignal, with ampdist.rvs(). For now sufficient


    eleSig, uncertainty_ele = esim.simulateElectronics(pmtSig, uncertainty_pmt, times)

    # adc signal
    stimes, samples, samples_unpro, uncertainty_sampled = esim.simulateADC(times, eleSig, uncertainty_ele, 1)

    #This part should be done all the time, even when it is loaded
    bl_mean, s_mean, std, std_unpro, bl_mean_uncertainty, bl_array, stddev_uncert_mean, stddev_mean, spike, skew = esim.FPGA(stimes, samples, samples_unpro, uncertainty_sampled, 1, True)


    plt.plot(stimes, samples, label="gain=2, duration=1e5 ns")
    plt.plot(stimes, bl_array, label="baseline array")
    plt.plot(stimes, np.repeat(stddev_mean+s_mean, len(stimes)), label="standard dev")
    plt.plot(stimes, np.repeat(bl_mean, len(stimes)), label="baseline mean")

plt.xlabel("time ns")
plt.ylabel("signal")
plt.legend(loc="upper right")

plt.figure()

for times in range(5):

    

    esim = TraceSimulation(
        ampSpec="../data/spe_R11920-RM_ap0.0002.dat",
        timeSpec="../data/bb3_1700v_timing.txt",
        pulseShape="../data/pulse_FlashCam_7dynode_v2a.dat",
        background_rate = 3e9,
        gain=15,
        no_signal_duration = 1e5,
        noise=0.8,
    )

    evts_br, k_evts = esim.simulateBackground(evts)

    # pmt signal
    times, pmtSig, uncertainty_pmt = esim.simulatePMTSignal(evts_br, k_evts) #TODO : make uncertainty from the simulatePMTSignal, with ampdist.rvs(). For now sufficient


    eleSig, uncertainty_ele = esim.simulateElectronics(pmtSig, uncertainty_pmt, times)

    # adc signal
    stimes, samples, samples_unpro, uncertainty_sampled = esim.simulateADC(times, eleSig, uncertainty_ele, 1)

    #This part should be done all the time, even when it is loaded
    bl_mean, s_mean, std, std_unpro, bl_mean_uncertainty, bl_array, stddev_uncert_mean, stddev_mean, spike, skew = esim.FPGA(stimes, samples, samples_unpro, uncertainty_sampled, 1, True)


    plt.plot(stimes, samples, label="gain=15, duration=1e5 ns")
    plt.plot(stimes, bl_array,label="baseline array")
    plt.plot(stimes, np.repeat(stddev_mean+s_mean, len(stimes)), label="standard dev")
    plt.plot(stimes, np.repeat(bl_mean, len(stimes)), label="baseline mean")

plt.xlabel("time ns")
plt.ylabel("signal")

plt.legend(loc="upper right")



plt.show()


