#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import rv_histogram, randint, poisson, expon, exponnorm
from scipy.signal import resample
import scipy.integrate as integrate

import sys
import os
"""
p1 = os.path.abspath(__file__+"/../../")
sys.path.insert(0, p1+"\\debug_fcts")
sys.path.insert(0, p1+"\\simulation")
sys.path.insert(0, p1+"\\darkcounts")
sys.path.insert(0, p1+"\\pulser")
"""
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

plt.rcParams['text.usetex'] = True

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
    return ((calculate_A(s, l, m)**2*l)/2)*np.exp(l**2*s**2)


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

"""
plt.figure()
plt.plot(esim_init.ampSpec[0][0:int(0.15*len(esim_init.ampSpec[0]))], esim_init.ampSpec[1][0:int(0.15*len(esim_init.ampSpec[1]))])
plt.grid()
plt.xlabel("Amplitude",fontsize=15)
plt.ylabel("Probability density",fontsize=15)
plt.show()
"""


pulse = Pulser(step=esim_init.t_step, pulse_type="none")
evts = pulse.generate_all()


brlinspace = np.logspace(7,10,num=30)

gainlinspace = np.linspace(4,16,num=8)

noise_linspace = [0.8]
#noise_linspace = [0.8]

####Only one window

plt.figure()

for gain in gainlinspace:
    extracted_gains = []
    for background_rate_ in brlinspace:

        esim = TraceSimulation(
            ampSpec="../data/spe_R11920-RM_ap0.0002.dat",
            timeSpec="../data/bb3_1700v_timing.txt",
            #pulseShape="data/pulse_FlashCam_7dynode_v2a.dat",
            background_rate = background_rate_,
            gain=gain,
            no_signal_duration = 1e5,
            noise=0.8,
        )

        pulse = Pulser(step=esim.t_step, duration=esim.no_signal_duration, pulse_type="none")
        evts = pulse.generate_all()

        
        evts_br, k_evts = esim.simulateBackground(evts)

        # pmt signal
        times, pmtSig, uncertainty_pmt = esim.simulatePMTSignal(evts_br, k_evts) #TODO : make uncertainty from the simulatePMTSignal, with ampdist.rvs(). For now sufficient


        eleSig, uncertainty_ele = esim.simulateElectronics(pmtSig, uncertainty_pmt, times)


        # adc signal
        stimes, samples, samples_unpro, uncertainty_sampled = esim.simulateADC(times, eleSig, uncertainty_ele, 1)

        eta_exp = np.std(samples)**2/(np.mean(samples)-esim.offset)
        eta_th = (esim.ampStddev**2+1)*calculate_J2(esim_init.ps_sigma, esim_init.ps_lambda, 0)/esim_init.singePE_area_theoretical



        #g_prime_mode = ((np.std(samples)**2)/(np.mean(samples)-esim.offset))*(esim.singePE_area_theoretical/((esim.ampStddev**2+1)*calculate_J2(esim.ps_sigma, esim.ps_lambda, 0)))
        
        r = (eta_exp/eta_th)/gain
        extracted_gains.append(r)

    plt.semilogx(brlinspace, extracted_gains, label="G="+str(format(gain,".2f")))


plt.legend(loc="upper right",fontsize=12)
plt.xlabel(r"$f_{NSB}$[Hz]",fontsize=15)
plt.ylabel(r"$\frac{G'}{G}$",fontsize=15)
plt.grid()

"""
fig, ax = plt.subplots(1, 3,constrained_layout=True)

iter_ = 0

for noise in noise_linspace:
    
    for gain in gainlinspace:
        extracted_gains = []
        for background_rate_ in brlinspace:

            esim = TraceSimulation(
                ampSpec="../data/spe_R11920-RM_ap0.0002.dat",
                timeSpec="../data/bb3_1700v_timing.txt",
                #pulseShape="data/pulse_FlashCam_7dynode_v2a.dat",
                background_rate = background_rate_,
                gain=gain,
                no_signal_duration = 2e5,
                noise=noise,
            )

            pulse = Pulser(step=esim.t_step, duration=esim.no_signal_duration, pulse_type="none")
            evts = pulse.generate_all()

            
            evts_br, k_evts = esim.simulateBackground(evts)

            # pmt signal
            times, pmtSig, uncertainty_pmt = esim.simulatePMTSignal(evts_br, k_evts) #TODO : make uncertainty from the simulatePMTSignal, with ampdist.rvs(). For now sufficient


            eleSig, uncertainty_ele = esim.simulateElectronics(pmtSig, uncertainty_pmt, times)


            # adc signal
            stimes, samples, samples_unpro, uncertainty_sampled = esim.simulateADC(times, eleSig, uncertainty_ele, 1)

            eta_exp = np.std(samples)**2/(np.mean(samples)-esim.offset)
            eta_th = (esim.ampStddev**2+1)*calculate_J2(esim_init.ps_sigma, esim_init.ps_lambda, 0)/esim_init.singePE_area_theoretical



            #g_prime_mode = ((np.std(samples)**2)/(np.mean(samples)-esim.offset))*(esim.singePE_area_theoretical/((esim.ampStddev**2+1)*calculate_J2(esim.ps_sigma, esim.ps_lambda, 0)))
            
            r = (eta_exp/eta_th)/gain
            extracted_gains.append(r)

        ax[iter_].semilogx(brlinspace, extracted_gains, label="gain="+str(gain))
    ax[iter_].set_title("Noise = "+str(noise))

    ax[iter_].legend(loc="upper left")
    ax[iter_].set_xlabel("NSB rate")
    ax[iter_].set_ylabel("G'/G")

    iter_ += 1


"""
plt.show()