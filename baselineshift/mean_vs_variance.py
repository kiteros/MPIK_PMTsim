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
import scipy.special as sse

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

def calculate_J2_theoretical(s, l, m):
    ##For now this equation is false
    inte = calculate_A(s, l, m)**2*s*l**2*np.exp(l**2*s**2)/np.sqrt(np.pi)
    return inte


#Checking the theoretical relationship between mean and variance

esim_init = TraceSimulation(
    ampSpec="../data/spe_R11920-RM_ap0.0002.dat",
    timeSpec="../data/bb3_1700v_timing.txt",
    #pulseShape="../data/pulse_FlashCam_7dynode_v2a.dat",
    background_rate = 1e7,
    gain=3,
    no_signal_duration = 7e4,
    noise=0.8

)


pulse = Pulser(step=esim_init.t_step, pulse_type="none")
evts = pulse.generate_all()

br_linspace = np.logspace(6,9,num=4)

means_array = []
means_array_th = []

variance_array = []
variance_array_th = []

eta_array = []
eta_array_th = []

reconstructed_gain = []

relative_gain = []

for br_rate in br_linspace:

    gain = 5


    esim = TraceSimulation(
        ampSpec="../data/spe_R11920-RM_ap0.0002.dat",
        timeSpec="../data/bb3_1700v_timing.txt",
        #pulseShape="../data/pulse_FlashCam_7dynode_v2a.dat",
        background_rate = br_rate,
        gain=gain,
        no_signal_duration = 2e5,
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

    means_array.append(np.mean(samples))
    th = gain * br_rate * esim.singePE_area_theoretical * 1e-9 + esim.offset
    means_array_th.append(th)

    variance_array.append(np.std(samples)**2)
    th_variance = br_rate*gain**2*(esim.ampStddev**2+1)*calculate_J2(esim_init.ps_sigma, esim_init.ps_lambda, 0)*1e-9
    variance_array_th.append(th_variance)

    

    eta_exp = np.std(samples)**2/(np.mean(samples)-esim.offset)
    eta_array.append(eta_exp)
    eta_th = (esim.ampStddev**2+1)*calculate_J2(esim_init.ps_sigma, esim_init.ps_lambda, 0)/esim_init.singePE_area_theoretical
    eta_array_th.append(eta_th*gain)

    reconstructed_gain.append(eta_exp/eta_th)

    relative_gain.append((eta_exp/eta_th)/gain)



plt.figure()
plt.semilogx(br_linspace, means_array,label="experimental")
plt.semilogx(br_linspace, means_array_th, label="theoretical")
plt.title("Means")
plt.legend(loc="upper left")


####Now for the variance

print("J2", calculate_J2(esim_init.ps_sigma, esim_init.ps_lambda, 0))
print("J2 th", calculate_J2_theoretical(esim_init.ps_sigma, esim_init.ps_lambda, 0)) 

plt.figure()
plt.semilogx(br_linspace, variance_array,label="experimental")
plt.semilogx(br_linspace, variance_array_th, label="theoretical")
plt.semilogx(br_linspace, np.repeat(esim_init.noise, len(br_linspace)), label="noise")
plt.title("Variance")
plt.legend(loc="upper left")



plt.figure()
plt.semilogx(br_linspace, eta_array, label="measured")
plt.semilogx(br_linspace, eta_array_th, label="theoretical")
plt.semilogx(br_linspace, reconstructed_gain, label="reconstructed gain")
plt.title("eta (Var/Mean)")

plt.legend(loc="upper left")

plt.figure()
plt.semilogx(br_linspace, relative_gain, label="relative")
plt.title("relative gains")

plt.legend(loc="upper left")

plt.show()