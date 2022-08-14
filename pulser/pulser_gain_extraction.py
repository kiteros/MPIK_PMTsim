#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import rv_histogram, randint, poisson, expon, exponnorm
from scipy.signal import resample
import scipy.integrate as integrate

import sys
import os
sys.path.insert(0, '/home/jebach/Documents/flashcam/pmt-trace-simulation-master/PMTtraceSIM_draft/debug_fcts')
sys.path.insert(0, '/home/jebach/Documents/flashcam/pmt-trace-simulation-master/PMTtraceSIM_draft/simulation')
sys.path.insert(0, '/home/jebach/Documents/flashcam/pmt-trace-simulation-master/PMTtraceSIM_draft/darkcounts')
sys.path.insert(0, '/home/jebach/Documents/flashcam/pmt-trace-simulation-master/PMTtraceSIM_draft/baselineshift')

"""
print(__file__)
p1 = os.path.abspath(__file__+"/../../")
sys.path.insert(0, p1+"\\debug_fcts")
sys.path.insert(0, p1+"\\simulation")
sys.path.insert(0, p1+"\\darkcounts")
sys.path.insert(0, p1+"\\baselineshift")
"""


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

from scipy import signal
import scipy.special as sse

import matplotlib.patches as patches

def expnorm_normalized(x,l,s,m):
    """
    Fits an exponentially modified gaussian (EMG), normalized (A=1)

    Parameters
    ----------
    x - x-values array
    l - pulse parameter (tail)
    s - pulse parameter (pulse)
    m - position parameter

    Returns 
    ---
    EMG distribitoon y-array

    """

    f = 0.5*l*np.exp(0.5*l*(2*m+l*s*s-2*x))*sse.erfc((m+l*s*s-x)/(np.sqrt(2)*s)) 
    return f

def find_nearest(array,value):

    array = np.sort(array)  ###sort ascending
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1], len(array)-(idx-1)-1
    else:
        return array[idx], len(array)-idx-1

def erfcxinv(v):

    """
    Inverse complementary error function scaled

    Parameters
    ----------
    v - input value

    Returns 
    ---
    output value

    """

    x = np.linspace(-10000,10000, num=1000000)
    y = sse.erfcx(x)

    value, position = find_nearest(y, v)

    return x[position]

def find_mode(sigma, lamda, mu):
    """
    Find the mode of an EMG

    Parameters
    ----------
    sigma - pulse parameter
    lamda - pulse tail parameter
    mu - pulse position parameter

    Returns 
    ---
    mode

    """
    pos = mu-np.sqrt(2)*sigma*erfcxinv((1/(lamda*sigma))*np.sqrt(2/np.pi))+sigma**2*lamda
    return pos

def calculate_A(sigma, lamda, mu):
    """
    Fits the normalization coefficient A such that the maximum reached 1

    Parameters
    ----------
    sigma - pulse parameter
    lamda - pulse parameter
    mu - pulse position parameter

    Returns 
    ---
    A

    """
    A=1/(expnorm_normalized(find_mode(sigma,lamda,mu),lamda,sigma,mu))
    return A

def compute_I2(s, l, m, s_prime):
	##compute with the trapeze method this integral

	bounds = 1000

	A = calculate_A(s,l,m)

	mode = find_mode(s,l,m)

	m_prime = mode

	x = np.linspace(-bounds,bounds, num=10000)
	f = A*0.5*l*np.exp(0.5*l*(2*m+l*s*s-2*x))*sse.erfc((m+l*s*s-x)/(np.sqrt(2)*s))  # exponential gaussian
	g = (1/(s_prime*np.sqrt(2*np.pi)))*np.exp(-(1/2)*((x-m_prime)**2)/(s_prime**2))
	h = f**2*g

	"""
	plt.figure()
	plt.plot(x, f)
	plt.show()
	"""

	inte = integrate.cumtrapz(h, x)[-1]

	return inte

def compute_I1(s, l, m, pulse, offset_integral = 0):
	##compute with the trapeze method this integral

	bounds = 1000

	A = calculate_A(s,l,m)

	s_prime = pulse.pulse_std

	x = np.linspace(-bounds,bounds, num=100000)
	f1 = A*0.5*l*np.exp(0.5*l*(2*m+l*s**2-2*(offset_integral-x)))*sse.erfc((m+l*s**2-(offset_integral-x))/(np.sqrt(2)*s))  # exponential gaussian
	g = (1/(s_prime*np.sqrt(2*np.pi)))*np.exp(-(1/2)*((x)**2)/(s_prime**2))
	#h = g*(f1+f2+f3)
	h = g*f1

	"""
	plt.figure()
	plt.plot(x, h)
	plt.show()
	"""

	inte = integrate.cumtrapz(h, x)[-1]

	###We have this empricial coefficient :

	#inte = inte*1.1

	return inte


def expected_value(s, l, m, pulse):

	bounds = 1000
	A = calculate_A(s,l,m)
	x = np.linspace(-bounds,bounds, num=10000)

	n_trials = 1000
	results = []
	for i in range(n_trials):

		t = -norm.rvs(loc=0, scale=pulse.pulse_std, size=1)

		t = A*0.5*l*np.exp(0.5*l*(2*m+l*s*s-2*t))*sse.erfc((m+l*s*s-t)/(np.sqrt(2)*s))

		results.append(t)

	return np.mean(results)

def calculate_I3(s, l, m, pulse):
	bounds = 1000
	x = np.linspace(-bounds,bounds, num=10000)

	A = calculate_A(s,l,m)
	mode = find_mode(s,l,m)

	period = (1.0/pulse.max_frequency)*(1e9) #In ns

	m_prime = mode#Maybe put it at 0 for this one
	s_prime = pulse.pulse_std

	

	f1 = A*0.5*l*np.exp(0.5*l*(2*m+l*s*s-2*(period/2 - x)))*sse.erfc((m+l*s*s-(period/2 - x))/(np.sqrt(2)*s))  # exponential gaussian
	f2 = A*0.5*l*np.exp(0.5*l*(2*m+l*s*s-2*(3*period/2- x)))*sse.erfc((m+l*s*s-(3*period/2 - x))/(np.sqrt(2)*s))  # exponential gaussian
	f3 = A*0.5*l*np.exp(0.5*l*(2*m+l*s*s-2*(5*period/2- x)))*sse.erfc((m+l*s*s-(5*period/2 - x))/(np.sqrt(2)*s))  # exponential gaussian
	f4 = A*0.5*l*np.exp(0.5*l*(2*m+l*s*s-2*(7*period/2- x)))*sse.erfc((m+l*s*s-(7*period/2 - x))/(np.sqrt(2)*s))  # exponential gaussian
	g = (1/(s_prime*np.sqrt(2*np.pi)))*np.exp(-(1/2)*((x-m_prime)**2)/(s_prime**2))

	h = g*(f1+f2+f3+f4)
	inte = integrate.cumtrapz(h, x, initial=0)[-1]

	return inte

def calculate_J2(s, l, m):
    bounds = 10000

    A = calculate_A(s,l,m)

    mode = find_mode(s,l,m)

    m_prime = mode

    x = np.linspace(-bounds,bounds, num=10000)
    f = A*0.5*l*np.exp(0.5*l*(2*m+l*s*s-2*x))*sse.erfc((m+l*s*s-x)/(np.sqrt(2)*s))  # exponential gaussian
    h = f**2

    """
    plt.figure()
    plt.plot(x, f)
    plt.show()
    """

    inte = integrate.cumtrapz(h, x)[-1]
    return inte

def find_h(s, l, m, s_prime):
	bounds_x = 1000
	bounds_t = 1000

	x_trials = np.linspace(-bounds_x, bounds_x, num=5000)

	minimizing_Eh = 0
	minimizing_Xh = 0



	for x in x_trials:

		t = np.linspace(-bounds_t,bounds_t, num=5000)
		f = 0.5*l*np.exp(0.5*l*(l*s*s-2*(x-t)))*sse.erfc((l*s*s-(x-t))/(np.sqrt(2)*s))  # exponential gaussian
		g = (1/(s_prime*np.sqrt(2*np.pi)))*np.exp(-(1/2)*((t)**2)/(s_prime**2))
		h = f*g

		inte = integrate.cumtrapz(h, t)[-1]

		if inte > minimizing_Eh:
			minimizing_Eh = inte
			minimizing_Xh = x

	print("Best minimizing x : ", minimizing_Xh)
	print("best minimizing integral value", minimizing_Eh)

	return minimizing_Xh


####start by loading a pulser and making it act on the trace, print it

gain_linspace = np.linspace(4,16,num=5)

standard_devs_peaks = []
mean_peaks = []
eta_peaks = []

mean_peaks_PMTsig = []
mean_peaks_ADC = []

mean_peaks_crests = []

esim_init = TraceSimulation(
    #ampSpec="../data/spe_R11920-RM_ap0.0002.dat",
    timeSpec="../data/bb3_1700v_timing.txt",
    #pulseShape="data/pulse_FlashCam_7dynode_v2a.dat",
    background_rate = 1e9,
    gain=10,
    no_signal_duration = 1e5,
    noise=1,
)




pulser_init = Pulser(step=esim_init.t_step, duration=esim_init.no_signal_duration, pulse_type="pulsed")
print("pulser std", pulser_init.pulse_std)

I_1 = compute_I1(esim_init.ps_sigma, esim_init.ps_lambda, esim_init.ps_mu, pulser_init)


for i in gain_linspace:

	esim = TraceSimulation(
	    #ampSpec="../data/spe_R11920-RM_ap0.0002.dat",
	    timeSpec="../data/bb3_1700v_timing.txt",
	    #pulseShape="data/pulse_FlashCam_7dynode_v2a.dat",
	    background_rate = 1e9,
	    gain=i,
	    no_signal_duration = 5e4,
	    noise=0.8,
	)

	pulse = Pulser(step=esim.t_step,freq=5e6, duration=esim.no_signal_duration, pulse_type="pulsed")
	evts = pulse.generate_all()

	
	evts_br, k_evts = esim.simulateBackground(evts)

	# pmt signal
	times, pmtSig, uncertainty_pmt = esim.simulatePMTSignal(evts_br, k_evts) #TODO : make uncertainty from the simulatePMTSignal, with ampdist.rvs(). For now sufficient


	maxs, _ = signal.find_peaks(pmtSig, prominence=1) ###promeminence devrait dependre du gain ? Non
	max_values_stimes_PMTsig = times[maxs]
	max_values_PMTsig = pmtSig[maxs]

	eleSig, uncertainty_ele = esim.simulateElectronics(pmtSig, uncertainty_pmt, times)

	maxs, _ = signal.find_peaks(eleSig, prominence=4) ###promeminence devrait dependre du gain ? Non
	max_values_stimes_eleSig = times[maxs]
	max_values_eleSig = eleSig[maxs]


	# adc signal
	stimes, samples, samples_unpro, uncertainty_sampled = esim.simulateADC(times, eleSig, uncertainty_ele, 1)

	#bl_mean, _, _, _, _, _, _, _, _, _ = esim.FPGA(stimes, samples, samples_unpro, uncertainty_sampled, 1, True)


	"""
	plt.figure()
	plt.plot(times, pmtSig)

	plt.figure()
	plt.plot(times, eleSig)
	plt.show()

	"""

	
	print("#####elements")
	print(esim_init.t_step)
	print(esim_init.pulseShape[0][2]-esim_init.pulseShape[0][1])
	print(times[3]-times[2])
	
	


	###Now the goal is to identify every peak

	##Use the upsampling here




	#samples = signal.resample(samples, 4*len(stimes), window='boxcar')
	#stimes = np.linspace(np.min(stimes), np.max(stimes),4*len(stimes), endpoint=False)

	dt_resampling = stimes[2]-stimes[1]


	maxs, _ = signal.find_peaks(samples, prominence=10) ###promeminence devrait dependre du gain ? Non
	max_values_stimes = stimes[maxs]
	max_values = samples[maxs]

	th_average = i*(pulser_init.pe_intensity*I_1)+esim_init.offset+esim_init.singePE_area*esim_init.background_rate*1e-9*i
	print("gain", i, th_average)
	#print("baseline", i, bl_mean)

	

	###let's use say the 2nd or 3rd peak as calibration


	######A good way to do a simple calibration is for example to take the highest peak in the 10 percent of the signal
	cal_index = np.argmax(samples[0:int(len(samples)*0.1)])
	stimes_cal = stimes[cal_index]


	#calibration = (max_values_stimes[5]+max_values_stimes[7]+max_values_stimes[9])/3

	###Print les 3 valeurs de calibration pour comprendre

	
	
	plt.figure()
	plt.plot(stimes, samples)

	plt.vlines(stimes_cal,200,600, color="g")

	calibration = stimes_cal


	#plt.scatter(max_values_stimes[5], max_values[5], marker='o', color='green')
	#plt.scatter(max_values_stimes[7], max_values[7], marker='o', color='green')
	#plt.scatter(max_values_stimes[9], max_values[9], marker='o', color='green')

	period = (1.0/pulse.max_frequency)*(1e9) #In ns
	number_pulses = int(pulse.duration // period)

	#width_region = pulse.pulse_std+period*pulse.pulse_to_pulse_jitter+period*0.3##last number is empricial
	width_region = period

	maximums_stimes = []
	maximums_values = []

	for i in range(int(0.7*number_pulses)):##Making sure we dont overshoot the signal
		i=i+1

		upper_bound = calibration+i*period-stimes[0]+width_region/2
		lower_bound = calibration+i*period-stimes[0]-width_region/2

		upper_bound_n_sample = int(upper_bound//dt_resampling)
		lower_bound_n_sample = int(lower_bound//dt_resampling)

		#print(calibration)
		#print(i)

		#plt.vlines(stimes[int((calibration+i*period-stimes[0])/dt_resampling)], 200, 600)##every peak
		#plt.vlines(calibration+i*period, 200, 600)
		#plt.vlines(stimes[upper_bound_n_sample], 200, 600)

		###shade the region

		
		plt.axvspan(stimes[lower_bound_n_sample], stimes[upper_bound_n_sample], alpha=0.5, color='red')



		#print(samples[lower_bound_n_sample:upper_bound_n_sample])


		max_index = np.argmax(samples[lower_bound_n_sample:upper_bound_n_sample])+lower_bound_n_sample

		maximums_values.append(samples[max_index])
		maximums_stimes.append(stimes[max_index])

	plt.scatter(maximums_stimes, maximums_values, marker='o', color='red')
	plt.xlabel("time in ns" + str(i))
	plt.ylabel("LSB")
	plt.close()
	
	

	

	
	
	


	###Do an histogram of the peaks
	hist, bins = np.histogram(max_values, density=True, bins=30)
	width = (bins[1] - bins[0])
	center = (bins[:-1] + bins[1:]) / 2

	
	"""
	plt.figure()
	plt.bar(center, hist, align='center', width=width)
	plt.title("gain : " + str(i))
	"""
	

	#standard_devs_peaks.append(np.std(max_values))
	#mean_peaks.append(np.mean(max_values))

	##Now we do it with the new peak finding method

	##below method with the period
	mean_peaks.append(np.mean(maximums_values))
	standard_devs_peaks.append(np.std(maximums_values))

	mean_peaks_PMTsig.append(np.mean(max_values_PMTsig))
	mean_peaks_ADC.append(np.mean(max_values_eleSig))


	print("std maxvalues", np.std(maximums_values))
	print("mean maxvalues", np.mean(maximums_values))
	eta_peaks.append(np.std(max_values)**2/(np.mean(max_values)-esim.offset-10))##Also removing the prominence



	###Find the distribution of the crests

	maxs_crest, _ = signal.find_peaks(-samples, prominence=2) ###promeminence devrait dependre du gain ? Non
	max_values_stimes_crest = stimes[maxs_crest]
	max_values_crest = -samples[maxs_crest]


	"""
	
	plt.figure()
	plt.plot(stimes, -samples)
	plt.scatter(max_values_stimes_crest, max_values_crest, marker='o', color='red')
	plt.xlabel("Number of samples" + str(i))
	plt.ylabel("LSB")
	"""
	

	###Do an histogram of the peaks
	hist_crests, bins_crests = np.histogram(max_values_crest, density=True, bins=30)
	width_crest = (bins_crests[1] - bins_crests[0])
	center_crest = (bins_crests[:-1] + bins_crests[1:]) / 2

	"""
	plt.figure()
	plt.bar(center_crest, hist_crests, align='center', width=width_crest)
	plt.title("crest, gain : " + str(i))
	"""
	
	mean_peaks_crests.append(-np.mean(max_values_crest)) ###important to put the minus to reverse again




plt.figure()
plt.plot(gain_linspace, standard_devs_peaks, label="measured std")

#############Let's comput the theoretical standard deviation

I_2 = compute_I2(esim_init.ps_sigma, esim_init.ps_lambda, esim_init.ps_mu, pulser_init.pulse_std)
theoretical_variance = pulser_init.pe_intensity*gain_linspace**2*(esim_init.ampStddev**2+1)*I_2+esim_init.background_rate*gain_linspace**2*(esim_init.ampStddev**2+1)*calculate_J2(esim_init.ps_sigma, esim_init.ps_lambda, 0)*1e-9


plt.plot(gain_linspace, np.sqrt(theoretical_variance), label="Theoretical STD")


####fit it
z = np.polyfit(gain_linspace, standard_devs_peaks, 1)
plt.plot(gain_linspace, [z[0]*x+z[1] for x in gain_linspace], label="Fit on measured STD")


plt.legend(loc="upper left")


plt.figure()
plt.plot(gain_linspace, mean_peaks)


theoretical_mean = calculate_A(esim_init.ps_sigma, esim_init.ps_lambda, 0)* gain_linspace*(pulser_init.pe_intensity*0.027911767902802007)+esim_init.offset+esim_init.singePE_area*esim_init.background_rate*1e-9*gain_linspace ###we add the prominence as an offset

plt.plot(gain_linspace, theoretical_mean, label="mean expected value")
plt.legend(loc="upper left")


#G=eta*coeff
#coeff = (var_th/mean_th)^-1

extracted_coeff = 1/((pulser_init.pe_intensity*(esim_init.ampStddev**2+1)*I_2
	+esim_init.background_rate*(esim_init.ampStddev**2+1)*calculate_J2(esim_init.ps_sigma, esim_init.ps_lambda, 0)*1e-9)/(calculate_A(esim_init.ps_sigma, esim_init.ps_lambda, 0)*(pulser_init.pe_intensity*0.027911767902802007)+esim_init.singePE_area*esim_init.background_rate*1e-9))

print("coef", extracted_coeff)
####do the same thing with the ratio

eta = []
eta_th = []

gain_extracted = []

for i in range(len(mean_peaks)):
	eta.append((standard_devs_peaks[i]**2)/(mean_peaks[i]-esim_init.offset))
	eta_th.append((theoretical_variance[i])/(theoretical_mean[i]-esim_init.offset))

print(eta)
plt.figure()
plt.plot(gain_linspace, eta, label="measured eta")
plt.plot(gain_linspace, eta_th)

plt.legend(loc="upper left")


plt.figure()
plt.plot(gain_linspace, gain_linspace)
plt.plot(gain_linspace, [x*extracted_coeff for x in eta], label="extracted gain")
plt.legend(loc="upper left")
plt.show()