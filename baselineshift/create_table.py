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

from scipy import special

def expnorm(x,l,s,m):

	f = 0.5*l*np.exp(0.5*l*(2*m+l*s*s-2*x))*special.erfc((m+l*s*s-x)/(np.sqrt(2)*s)) 
	return f

def find_nearest(array,value):

	array = np.sort(array)  ###sort ascending
	idx = np.searchsorted(array, value, side="left")
	if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
		return array[idx-1], len(array)-(idx-1)-1
	else:
		return array[idx], len(array)-idx-1

def erfcxinv(v):

	x = np.linspace(-10000,10000, num=1000000)
	y = special.erfcx(x)

	value, position = find_nearest(y, v)

	return x[position]

def find_mode(sigma, lamda, mu):
	pos = mu-np.sqrt(2)*sigma*erfcxinv((1/(lamda*sigma))*np.sqrt(2/np.pi))+sigma**2*lamda
	return pos

def calculate_A(sigma, lamda, mu):
	A=1/(expnorm(find_mode(sigma,lamda,mu),lamda,sigma,mu))
	return A


def calculate_eta(lamda, sigma, mu):
	tau = 2.821327162945735

	eta = tau * calculate_A(sigma, lamda, mu) * lamda *(1/4) * np.exp(lamda**2*sigma**2)*special.erfc(lamda*sigma)
	return eta 


line_numbers = []
#####Loop on all the files to gather all the line numbers
for filename in os.listdir('../exports/'):
	if ( not(filename.split("line=")[1].split(".")[0] in line_numbers)):
		line_numbers.append(filename.split("line=")[1].split(".")[0])

print("All lines", line_numbers)


for line in line_numbers:
	with open('../tables/table_traces='+line+'.csv', 'w', newline='') as csvfile:

		spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
		spamwriter.writerow(["True NSB Background", "True gain", "ADC noise", "N samples", "lamda poisson", "amp std", "pulse shape total sum", "ps_sigma", "ps_lamda","eta", "Baseline mean", "Offset", "Baseline mean - offset", 
			"Signal Variance", "Signal mean", "Baseline/Var", "Var/(Mean - Offset)"])

		for filename in os.listdir('../exports/'):
		    f = os.path.join('../exports/', filename)
		    # checking if it is a file
		    if os.path.isfile(f):
		        #Just add to an array

		        if filename.split("line=")[1].split(".")[0] == line:

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

			            ps_sigma = np.load(file)
			            ps_lambda = np.load(file)

			            N_samples = np.load(file)

			            lamda_poisson = np.load(file)
			            amp_std = np.load(file)
			            pulseShape_totalSum = np.load(file)


			            ratio = (stddev_mean**2)/bl_mean
			            ratio_mean = 0

			            eta = calculate_eta(ps_lambda, ps_sigma, 0)

			            spamwriter.writerow([true_background_rate, true_gain, adc_noise, N_samples, lamda_poisson, amp_std, pulseShape_totalSum, ps_sigma, ps_lambda, eta, bl_mean, 0, 0, stddev_mean**2, s_mean, ratio, ratio_mean])




for line in line_numbers:
	# assign dataset
	data = pd.read_csv('../tables/table_traces='+line+'.csv')   


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
		


	data.to_csv('../tables/table_traces='+line+'.csv')