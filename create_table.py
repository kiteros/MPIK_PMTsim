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

from debug_fcts.bl_shift import BL_shift
from debug_fcts.bl_stddev import BL_stddev
from debug_fcts.under_c import Under_c
from debug_fcts.debug import Debug
#from debug_fcts.baseline import Baseline
from debug_fcts.pulse import Pulse

from scipy.stats import norm

from mpl_toolkits.mplot3d import axes3d, Axes3D
from sklearn import linear_model 

from calculate_gains import GainCalculator
import csv


# importing pandas package
import pandas as pd


with open('table_traces.csv', 'w', newline='') as csvfile:

	spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
	spamwriter.writerow(["True NSB Background", "True gain", "ADC noise", "Baseline mean", "Offset", "Baseline mean - offset", 
		"Signal Variance", "Signal mean", "Baseline/Var", "Var/(Mean - Offset)"])

	for filename in os.listdir('exports/'):
	    f = os.path.join('exports/', filename)
	    # checking if it is a file
	    if os.path.isfile(f):
	        #Just add to an array

	        if filename.split("line=")[1].split(".")[0] == str(1):

		        with open(f, 'rb') as file:

		            stimes = np.load(file)
		            samples = np.load(file)
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

		            ratio = (stddev_mean**2)/bl_mean
		            ratio_mean = 0

		            spamwriter.writerow([true_background_rate, true_gain, adc_noise, bl_mean, 0, 0, stddev_mean**2, s_mean, ratio, ratio_mean])




  
# assign dataset
data = pd.read_csv('table_traces.csv')   


##rerun through the table, and look at the smallest nsb for each gain, and update the offset with that 


# sort data frame
data.sort_values(["True gain"], axis=0, ascending=[False], inplace=True)



current_gain = 15
smallest = 10000000000000
smallest_value = 0

smallest_values = []

for ch in data.iterrows():

	if ch[1]['True gain'] != current_gain:


		####add columns
		smallest_values.append(smallest_value)

		current_gain = ch[1]['True gain']
		smallest = 100000000000000
		smallest_value = 0

	if ch[1]['Baseline mean'] < smallest:
		smallest = ch[1]['Baseline mean']
		smallest_value = ch[1]['Baseline mean']


	

smallest_values.append(smallest_value)

print(smallest_values)

current_gain = 15
keeper = 0

print(len(smallest_values))

row = 0

for index, ch in data.iterrows():
	print(keeper)

	if ch['True gain'] != current_gain:
		keeper += 1
		current_gain = ch['True gain']
	#ch[1]["Offset"] = smallest_values[keeper]

	data.loc[index, "Offset"] = smallest_values[keeper]
	data.loc[index, "Baseline mean - offset"] = ch['Baseline mean'] - smallest_values[keeper]


	data.loc[index, "Baseline/Var"] = ch['Signal Variance']/(ch['Baseline mean'] - smallest_values[keeper])
	data.loc[index, "Var/(Mean - Offset)"] = ch['Signal Variance']/(ch["Signal mean"] - 199.0)
	


data.to_csv('table_traces.csv')