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
import scipy.fftpack

from scipy import special


###################
#Here we create a signal with different sample (length) and measure the decrease of standard variation uncertainty with sample size
#In a way that is the standard deviation of the standard deviation 
###############


def get_baseline_uncertainty(N, sigma_PE):
	#N: number of samples
	#sigma_PE : uncertainty of one point

	####Array for les P+,i car P-,i=1-P+,i
	n = N
	P_plus = 0.5*np.ones(n+1)
	baseline_speed = 0.125

	##now P_plus depends on k
	for i in range(len(P_plus)):
		P_plus[i] = 1-scipy.stats.norm(0, sigma_PE).cdf(i*baseline_speed)




	nb_k = 2*n-1

	S = np.zeros((nb_k,n))
	middle_index = int((nb_k-1)/2)
	S[middle_index][0] = 1


	###calculate the standard deviation at every n
	standard_deviation = []
	standard_deviation.append(0) ###for n=0


	for i in range(n-1):
		i = i+1 ##on saute l'index 0


		###So here we are at k = 1
		#we have k+1 nodes, starting at k-1 from the previous one, incrementing by two
		for k in range(i+1):
			current_k = middle_index-i+2*k

			absolute_k = abs(current_k-middle_index)
			

			if k == 0:
				####premier element de la ligne
				##sera toujours en absolute k negatif
				S[current_k][i] = S[current_k+1][i-1]*P_plus[absolute_k-1]
			elif k == i:
				##dernier element de la ligne
				S[current_k][i] = S[current_k-1][i-1]*P_plus[absolute_k-1]
			else :

				###Ligne dapres only pour ceux du milieu
				if current_k-middle_index>0:

					S[current_k][i] = S[current_k+1][i-1]*(1-P_plus[absolute_k+1])+S[current_k-1][i-1]*P_plus[absolute_k-1]
				elif current_k-middle_index == 0:
					S[current_k][i] = S[current_k+1][i-1]*(1-P_plus[1])+S[current_k-1][i-1]*(1-P_plus[1])
				else:
					S[current_k][i] = S[current_k+1][i-1]*(P_plus[absolute_k-1])+S[current_k-1][i-1]*(1-P_plus[absolute_k+1])

			this_row = [x[i] for x in S]


		#we need to know the value of the 
		esp_x = 0
		esp_x2 = 0
		for k in range(i+1):
			current_k = middle_index-i+2*k
			non_absolute_k = current_k-middle_index
			esp_x += this_row[current_k]*non_absolute_k*baseline_speed
			esp_x2 += this_row[current_k]*(non_absolute_k*baseline_speed)**2

		std = np.sqrt(esp_x2-esp_x**2)
		
		standard_deviation.append(std)

		print("calculating row...", i)
		



	"""
	last_row = [x[-1] for x in S]


	###remove the 0
	last_row = [i for i in last_row if i != 0]

	plt.figure()
	plt.plot(last_row)
	plt.show()
	"""

	standard_deviation_baseline_mean = 0
	for x in standard_deviation:
		standard_deviation_baseline_mean += x**2

	standard_deviation_baseline_mean = (1/n)*np.sqrt(standard_deviation_baseline_mean)

	std_fixed = (1/np.sqrt(n))*standard_deviation[-1]
	return standard_deviation_baseline_mean, std_fixed

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


def get_theoretical_gain(stddev, lamda, sigma, mu, baseline_mean, offset):
	
	tau = 2.821327162945735

	T_g = (4*stddev**2)/(tau*calculate_A(sigma,lamda,mu)*lamda*np.exp(sigma**2*lamda**2)*special.erfc(sigma*lamda)*(baseline_mean-offset))

	return T_g

def get_eta(lamda, sigma, mu):
	tau = 2.821327162945735
	eta = tau*calculate_A(sigma,lamda,mu)*lamda*np.exp(sigma**2*lamda**2)*special.erfc(sigma*lamda)/4
	return eta


esim_init = TraceSimulation(
    ampSpec="../data/spe_R11920-RM_ap0.0002.dat",
    timeSpec="../data/bb3_1700v_timing.txt",
    pulseShape="../data/pulse_FlashCam_7dynode_v2a.dat",
    background_rate = 1e7,
    gain=3,
    no_signal_duration = 7e4,
    noise=0.8

)


pulse = Pulser(step=esim_init.t_step, pulse_type="none")
evts = pulse.generate_all()

gain_standard_deviation = []
gain_standard_deviation_theoretical = []

gain_theoretical_current_std = []
gain_current = []

trace_lenght = []
nb_samples = []



for length in np.logspace(2,6, num=5):


	for times in range(10):

		offset = 199.54005208333334


		time = 0
		if length > 1e4:
			time = length
		else:
			time = 1e4

		esim = TraceSimulation(
	        ampSpec="../data/spe_R11920-RM_ap0.0002.dat",
	        timeSpec="../data/bb3_1700v_timing.txt",
	        #pulseShape="data/pulse_FlashCam_7dynode_v2a.dat",
	        background_rate = 1e7,
	        gain=3,
	        no_signal_duration = time,
	        noise=0.8,

	    )

		



		evts_br, k_evts = esim.simulateBackground(evts)

		times, pmtSig, uncertainty_pmt = esim.simulatePMTSignal(evts_br, k_evts)
		eleSig, uncertainty_ele = esim.simulateElectronics(pmtSig, uncertainty_pmt, times)
		stimes, samples, samples_unpro, uncertainty_sampled = esim.simulateADC(times, eleSig, uncertainty_ele, 1)
		bl_mean, s_mean, std, std_unpro, bl_mean_uncertainty, bl_array, stddev_uncert_mean, stddev_mean, spike, skew = esim.FPGA(stimes, samples, samples_unpro, uncertainty_sampled, 1, True)
		stimes = stimes[100:100+int(length)]
		samples = samples[100:100+int(length)]

		extracted_gain = get_theoretical_gain(np.std(samples), esim.ps_lambda, esim.ps_sigma, esim.ps_mu, bl_mean, offset)

		nb_samp = length/esim.t_sample

		print("nbsamp",nb_samp)

		sigm_PE = np.sqrt(esim.ampStddev**2*(esim.lamda+esim.lamda**2)+esim.lamda)*np.sqrt(esim.pulseShape_TotalSomation)*esim.gain+esim.noise
		#standard_dev = np.sqrt((2*sigm_PE**2/(get_eta(esim.ps_lambda, esim.ps_sigma, esim.ps_mu)*np.sqrt(nb_samp)*(bl_mean-offset)))**2+((stddev_mean**2*get_baseline_uncertainty(int(nb_samp), sigm_PE)[0])/(get_eta(esim.ps_lambda, esim.ps_sigma, esim.ps_mu)*(bl_mean-offset)))**2)
		standard_dev = 2*sigm_PE**2/(get_eta(esim.ps_lambda, esim.ps_sigma, esim.ps_mu)*np.sqrt(nb_samp)*(bl_mean-offset))


		gain_theoretical_current_std.append(standard_dev)

		gain_current.append(extracted_gain)

	    

	trace_lenght.append(length)

	nb_samp = length/esim.t_sample
	nb_samples.append(nb_samp)
	gain_standard_deviation.append(np.std(gain_current))
	gain_current = []

	gain_standard_deviation_theoretical.append(np.mean(gain_theoretical_current_std))
	gain_theoretical_current_std = []

	

plt.figure()
plt.semilogx(nb_samples, gain_standard_deviation, label="True")
plt.semilogx(nb_samples, gain_standard_deviation_theoretical, label="theoretical")
plt.xlabel("Number of samples")
plt.ylabel("Standard deviation of extracted gain")
plt.legend(loc="upper left")
plt.show()

