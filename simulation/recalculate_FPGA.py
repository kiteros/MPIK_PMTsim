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

#Amplitude spectrum obtained from spe_R11920-RM_ap0.0002.dat
ampSpec = np.loadtxt("../data/bb3_1700v_spe.txt", unpack=True)
timeSpec = "../data/bb3_1700v_timing.txt"
pulseShape = np.loadtxt("../data/pulse_FlashCam_7dynode_v2a.dat", unpack=True)

# init class
esim_init = TraceSimulation(
    ampSpec="../data/spe_R11920-RM_ap0.0002.dat",
    timeSpec="../data/bb3_1700v_timing.txt",
    pulseShape="../data/pulse_FlashCam_7dynode_v2a.dat",
)


counter = 0

for filename in os.listdir('../exports/'):
    f = os.path.join('../exports/', filename)
    # checking if it is a file
    if os.path.isfile(f):
        #Just add to an array
        print("updating...", counter)
        counter+=1

        if filename.split("line=")[1].split(".")[0] == str(1):

        	bl_mean = 0

	        with open(f, 'rb') as file:

	            stimes = np.load(file)
	            samples = np.load(file)
	            
	            ###Do not use the ones below
	            samples_unpro = np.zeros(samples.shape)
	            uncertainty_sampled = np.load(file)
	            bl_mean = np.load(file)
	            std = np.load(file)
	            stddev_mean = np.load(file)
	            spike = np.load(file)
	            s_mean = np.load(file)
	            std_unpro = np.load(file)
	            bl_mean_uncertainty = np.load(file)
	            bl_array = np.load(file)
	            stddev_uncert_mean = np.load(file)
	            skew = np.load(file)
	            adc_noise = np.load(file)
	            true_gain = np.load(file)
	            true_background_rate = np.load(file)

	            ##Recalculate FPGA

	            bl = esim_init.singePE_area*true_gain*true_background_rate*1e-9+200 ###A remplacer par le offset theorique
	            bl_array = []

	            for i in range(len(stimes)):

	            	if samples[i] > bl:
	            		bl += 0.125

	            	elif samples[i] < bl:
	            		bl -= 0.125

	            	bl_array.append(bl)

	            bl_mean = statistics.fmean(bl_array)

	            ###Maybe recalculate the standard deviation
	            stddev_mean = np.std(samples, ddof=1)

	        with open(f, 'wb') as f:

	        	##Resave everything

	        	np.save(f, stimes)
	        	np.save(f, samples)
	        	np.save(f, uncertainty_sampled)
	        	np.save(f, bl_mean)
	        	np.save(f, std)
	        	np.save(f, stddev_mean)
	        	np.save(f, spike)
	        	np.save(f, s_mean)
	        	np.save(f, std_unpro)
	        	np.save(f, bl_mean_uncertainty)
	        	np.save(f, bl_array)
	        	np.save(f, stddev_uncert_mean)
	        	np.save(f, skew)
	        	np.save(f, adc_noise)
	        	np.save(f, true_gain)
	        	np.save(f, true_background_rate)
	            