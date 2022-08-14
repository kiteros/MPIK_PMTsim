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

def compute_I2(s, l, m, s_prime, offset_integral = 0):
	##compute with the trapeze method this integral

	bounds = 1000

	A = calculate_A(s,l,m)

	mode = find_mode(s,l,m)

	m_prime = mode

	x = np.linspace(-bounds,bounds, num=10000)
	f = A*0.5*l*np.exp(0.5*l*(2*m+l*s*s-2*(x-offset_integral)))*sse.erfc((m+l*s*s-(x-offset_integral))/(np.sqrt(2)*s))  # exponential gaussian
	g = (1/(s_prime*np.sqrt(2*np.pi)))*np.exp(-(1/2)*((x)**2)/(s_prime**2))
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
	mode = find_mode(s,l,m)

	s_prime = pulse.pulse_std
	period = (1.0/pulse.max_frequency)*(1e9) #In ns

	m_prime = mode
	print("mode", mode)
	off = 46
	x = np.linspace(-bounds,bounds, num=100000)
	f1 = A*0.5*l*np.exp(0.5*l*(2*m+l*s**2-2*(x-offset_integral)))*sse.erfc((m+l*s**2-(x-offset_integral))/(np.sqrt(2)*s))  # exponential gaussian
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


brlinspace = np.logspace(9,9,num=1)

gainlinspace = np.linspace(10,10,num=1)

noise_linspace = [0.8]
#noise_linspace = [0.8]

####Only one window

esim_init = TraceSimulation(
    #ampSpec="../data/spe_R11920-RM_ap0.0002.dat",
    timeSpec="../data/bb3_1700v_timing.txt",
    #pulseShape="data/pulse_FlashCam_7dynode_v2a.dat",
    background_rate = 1e6,
    gain=10,
    no_signal_duration = 1e5,
    noise=1,
)

pulser_init = Pulser(step=esim_init.t_step, duration=esim_init.no_signal_duration, pulse_type="pulsed")
I_2 = compute_I2(esim_init.ps_sigma, esim_init.ps_lambda, esim_init.ps_mu, pulser_init.pulse_std)


plt.figure()

for gain in gainlinspace:
    extracted_gains = []
    for background_rate_ in brlinspace:

        esim = TraceSimulation(
            #ampSpec="../data/spe_R11920-RM_ap0.0002.dat",
            timeSpec="../data/bb3_1700v_timing.txt",
            #pulseShape="data/pulse_FlashCam_7dynode_v2a.dat",
            background_rate = background_rate_,
            gain=gain,
            no_signal_duration = 1e5,
            noise=0.8,
        )

        pulse = Pulser(step=esim.t_step,freq=5e6, duration=esim.no_signal_duration, pulse_type="pulsed")
        evts = pulse.generate_all()

        

        
        evts_br, k_evts = esim.simulateBackground(evts)

        # pmt signal
        times, pmtSig, uncertainty_pmt = esim.simulatePMTSignal(evts_br, k_evts) #TODO : make uncertainty from the simulatePMTSignal, with ampdist.rvs(). For now sufficient


        eleSig, uncertainty_ele = esim.simulateElectronics(pmtSig, uncertainty_pmt, times)


        # adc signal
        stimes, samples, samples_unpro, uncertainty_sampled = esim.simulateADC(times, eleSig, uncertainty_ele, 1)

        dt_resampling = stimes[2]-stimes[1]

        cal_index = np.argmax(samples[0:int(len(samples)*0.1)])
        stimes_cal = stimes[cal_index]

        plt.figure()
        plt.plot(stimes, samples)
        plt.vlines(stimes_cal,200,600, color="g")
        calibration = stimes_cal

        period = (1.0/pulse.max_frequency)*(1e9)
        number_pulses = int(pulse.duration // period)

        width_region = period
        maximums_stimes = []
        maximums_values = []

        for i in range(int(0.7*number_pulses)):
        	i=i+1

        	upper_bound = calibration+i*period-stimes[0]+width_region/2
        	lower_bound = calibration+i*period-stimes[0]-width_region/2

        	upper_bound_n_sample = int(upper_bound//dt_resampling)
        	lower_bound_n_sample = int(lower_bound//dt_resampling)
        	plt.axvspan(stimes[lower_bound_n_sample], stimes[upper_bound_n_sample], alpha=0.5, color='red')
        	max_index = np.argmax(samples[lower_bound_n_sample:upper_bound_n_sample])+lower_bound_n_sample
        	maximums_values.append(samples[max_index])
        	maximums_stimes.append(stimes[max_index])

        plt.scatter(maximums_stimes, maximums_values, marker='o', color='red')
        plt.xlabel("time in ns" + str(background_rate_))
        plt.ylabel("LSB")
        
        plt.close()

        print("std maxvalues", np.std(maximums_values))

        print("mean maxvalues", np.mean(maximums_values))

        eta = np.std(maximums_values)**2/(np.mean(maximums_values)-esim.offset)
        print(eta)
        extracted_coeff = 1/((pulse.pe_intensity*(esim.ampStddev**2+1)*I_2
			+esim.background_rate*(esim.ampStddev**2+1)*calculate_J2(esim.ps_sigma, esim.ps_lambda, 0)*1e-9)/(calculate_A(esim.ps_sigma, esim.ps_lambda, 0)*(pulser_init.pe_intensity*0.027911767902802007)+esim.singePE_area*esim.background_rate*1e-9))

        print("coef", extracted_coeff)

        gain_extracted = eta*extracted_coeff
        extracted_gains.append(gain_extracted/gain)

    plt.semilogx(brlinspace, extracted_gains, label="G="+str(format(gain,".2f")))


plt.legend(loc="upper right",fontsize=12)
plt.xlabel(r"$f_{NSB}$[Hz]",fontsize=15)
plt.ylabel(r"$\frac{G'}{G}$",fontsize=15)
plt.grid()

plt.show()