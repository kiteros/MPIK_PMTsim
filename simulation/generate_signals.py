#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import rv_histogram, randint, poisson, expon, exponnorm
from scipy.signal import resample
import scipy.integrate as integrate

import sys
sys.path.insert(0, '/home/jebach/Documents/flashcam/pmt-trace-simulation-master/PMTtraceSIM_draft/debug_fcts')
sys.path.insert(0, '/home/jebach/Documents/flashcam/pmt-trace-simulation-master/PMTtraceSIM_draft/baselineshift')
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

for gain in np.linspace(1,20,num=5):
    for br in np.logspace(6.0, 11.0, num=9):

        esim = TraceSimulation(
            ampSpec="../data/spe_R11920-RM_ap0.0002.dat",
            timeSpec="../data/bb3_1700v_timing.txt",
            #pulseShape="../data/pulse_FlashCam_7dynode_v2a.dat",
            background_rate = br,
            gain=gain,
            no_signal_duration = 1e5,

            ps_mu = 15.11,
            ps_lambda = 0.0659,
            ps_sigma = 2.7118,
        )

        ###########check maybe with have an issue with offset (print the this)

        evts_br, k_evts = esim.simulateBackground(evts)
        times, pmtSig, uncertainty_pmt = esim.simulatePMTSignal(evts_br, k_evts)
        eleSig, uncertainty_ele = esim.simulateElectronics(pmtSig, uncertainty_pmt, times)
        stimes, samples, samples_unpro, uncertainty_sampled = esim.simulateADC(times, eleSig, uncertainty_ele, 1)
        bl_mean, s_mean, std, std_unpro, bl_mean_uncertainty, bl_array, stddev_uncert_mean, stddev_mean, spike, skew = esim.FPGA(stimes, samples, samples_unpro, uncertainty_sampled, 1, True)

        print("background rates", br)
        print(gain)