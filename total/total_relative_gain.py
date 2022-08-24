#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import rv_histogram, randint, poisson, expon, exponnorm
from scipy.signal import resample
import scipy.integrate as integrate

import sys
import os

sys.path.insert(0, '/home/jebach/Documents/flashcam/pmt-trace-simulation-master/PMTtraceSIM_draft/simulation')
sys.path.insert(0, '/home/jebach/Documents/flashcam/pmt-trace-simulation-master/PMTtraceSIM_draft/debug_fcts')
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
import scipy.fftpack

from scipy import special

from scipy.stats import norm
from sklearn.neighbors import KernelDensity

from scipy import signal

from pulse_peak_histogram_2 import PeakHistogram

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

### input_ = [N,G,Br]
### output_ = [P,W,R,K]

input_ = [0.8, 7, 316227]
output_ = [4.75, 1, 3, 3.75]

window_ = ''

if output_[1] == 0:
	window_ = 'tukey'
elif output_[1] == 1:
	window_ = 'blackman'



brlinspace = np.logspace(4,11,num=30)
brlinspace_darkcount = np.logspace(4,8.5,num=30)

gain=5


plt.figure()


###Darkcounts

extracted_gains = []
for background_rate_ in brlinspace_darkcount:
	ph = PeakHistogram(
		noise=0.8,
		background_rate=background_rate_,
		gain_linspace=[gain],
		graphs=False,
		verbose=False,
		prominence = output_[0],
	    window = window_,
	    resampling_rate=output_[2],
	    kde_banwidth=output_[3],
	    trace_lenght=1e5,

	)

	r = ph.get_relative_gain_array()[0]
	extracted_gains.append(r)

plt.semilogx(brlinspace_darkcount, extracted_gains, '--', label="Darkcounts, G=5")



####Let's do the others

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
    eta_th = (esim.ampStddev**2+1)*calculate_J2(esim.ps_sigma, esim.ps_lambda, 0)/esim.singePE_area_theoretical



    #g_prime_mode = ((np.std(samples)**2)/(np.mean(samples)-esim.offset))*(esim.singePE_area_theoretical/((esim.ampStddev**2+1)*calculate_J2(esim.ps_sigma, esim.ps_lambda, 0)))
    
    r = (eta_exp/eta_th)/esim.gain
    extracted_gains.append(r)

plt.semilogx(brlinspace, extracted_gains, label="Variance/Mean, G=5")


##########Now lets do the flasher

standard_devs_peaks = []
mean_peaks = []
eta_peaks = []

esim_init = TraceSimulation(
    #ampSpec="../data/spe_R11920-RM_ap0.0002.dat",
    timeSpec="../data/bb3_1700v_timing.txt",
    #pulseShape="data/pulse_FlashCam_7dynode_v2a.dat",
    background_rate = 1e9,
    gain=gain,
    no_signal_duration = 1e5,
    noise=1,
)


pulser_init = Pulser(step=esim_init.t_step, duration=esim_init.no_signal_duration, pulse_type="pulsed")
print("pulser std", pulser_init.pulse_std)

I_1 = compute_I1(esim_init.ps_sigma, esim_init.ps_lambda, esim_init.ps_mu, pulser_init)


for j in brlinspace:

    esim = TraceSimulation(
        #ampSpec="../data/spe_R11920-RM_ap0.0002.dat",
        timeSpec="../data/bb3_1700v_timing.txt",
        #pulseShape="data/pulse_FlashCam_7dynode_v2a.dat",
        background_rate = j,
        gain=gain,
        no_signal_duration = 1e5,
        noise=0.8,
    )

    pulse = Pulser(step=esim.t_step,freq=5e6, duration=esim.no_signal_duration, pulse_type="pulsed")
    evts = pulse.generate_all()

    
    evts_br, k_evts = esim.simulateBackground(evts)
    times, pmtSig, uncertainty_pmt = esim.simulatePMTSignal(evts_br, k_evts) #TODO : make uncertainty from the simulatePMTSignal, with ampdist.rvs(). For now sufficient
    eleSig, uncertainty_ele = esim.simulateElectronics(pmtSig, uncertainty_pmt, times)
    stimes, samples, samples_unpro, uncertainty_sampled = esim.simulateADC(times, eleSig, uncertainty_ele, 1)
    dt_resampling = stimes[2]-stimes[1]


    maxs, _ = signal.find_peaks(samples, prominence=10) ###promeminence devrait dependre du gain ? Non
    max_values_stimes = stimes[maxs]
    max_values = samples[maxs]



    ######A good way to do a simple calibration is for example to take the highest peak in the 10 percent of the signal
    cal_index = np.argmax(samples[0:int(len(samples)*0.1)])
    stimes_cal = stimes[cal_index]

    
    plt.figure()
    plt.plot(stimes, samples)

    plt.vlines(stimes_cal,200,600, color="g")

    calibration = stimes_cal


    period = (1.0/pulse.max_frequency)*(1e9) #In ns
    number_pulses = int(pulse.duration // period)

    width_region = period

    maximums_stimes = []
    maximums_values = []

    for i in range(int(0.7*number_pulses)):##Making sure we dont overshoot the signal
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
    plt.xlabel("time in ns" + str(i))
    plt.ylabel("LSB")
    plt.close()
    
    hist, bins = np.histogram(max_values, density=True, bins=30)
    width = (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2

    mean_peaks.append(np.mean(maximums_values))
    standard_devs_peaks.append(np.std(maximums_values))

    eta_peaks.append(np.std(max_values)**2/(np.mean(max_values)-esim.offset-10))##Also removing the prominence



I_2 = compute_I2(esim_init.ps_sigma, esim_init.ps_lambda, esim_init.ps_mu, pulser_init.pulse_std)
theoretical_variance = pulser_init.pe_intensity*gain**2*(esim_init.ampStddev**2+1)*I_2+esim_init.background_rate*gain**2*(esim_init.ampStddev**2+1)*calculate_J2(esim_init.ps_sigma, esim_init.ps_lambda, 0)*1e-9

theoretical_mean = calculate_A(esim_init.ps_sigma, esim_init.ps_lambda, 0)* gain*(pulser_init.pe_intensity*0.027911767902802007)+esim_init.offset+esim_init.singePE_area*esim_init.background_rate*1e-9*gain ###we add the prominence as an offset


extracted_coeff = 1/((pulser_init.pe_intensity*(esim_init.ampStddev**2+1)*I_2
    +esim_init.background_rate*(esim_init.ampStddev**2+1)*calculate_J2(esim_init.ps_sigma, esim_init.ps_lambda, 0)*1e-9)/(calculate_A(esim_init.ps_sigma, esim_init.ps_lambda, 0)*(pulser_init.pe_intensity*0.027911767902802007)+esim_init.singePE_area*esim_init.background_rate*1e-9))


eta = []
eta_th = []

gain_extracted = []

for i in range(len(mean_peaks)):
    eta.append((standard_devs_peaks[i]**2)/(mean_peaks[i]-esim_init.offset))
    #eta_th.append((theoretical_variance)/(theoretical_mean-esim_init.offset))




#plt.plot(gain_linspace, gain_linspace)
plt.semilogx(brlinspace, [x*extracted_coeff/gain for x in eta], ':', label="Pulser, G=5")



####second gain


gain=13




###Darkcounts

extracted_gains = []
for background_rate_ in brlinspace_darkcount:
    ph = PeakHistogram(
        noise=0.8,
        background_rate=background_rate_,
        gain_linspace=[gain],
        graphs=False,
        verbose=False,
        prominence = output_[0],
        window = window_,
        resampling_rate=output_[2],
        kde_banwidth=output_[3],
        trace_lenght=1e5,

    )

    r = ph.get_relative_gain_array()[0]
    extracted_gains.append(r)

plt.semilogx(brlinspace_darkcount, extracted_gains, '--', label="Darkcounts, G=13")



####Let's do the others

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
    eta_th = (esim.ampStddev**2+1)*calculate_J2(esim.ps_sigma, esim.ps_lambda, 0)/esim.singePE_area_theoretical



    #g_prime_mode = ((np.std(samples)**2)/(np.mean(samples)-esim.offset))*(esim.singePE_area_theoretical/((esim.ampStddev**2+1)*calculate_J2(esim.ps_sigma, esim.ps_lambda, 0)))
    
    r = (eta_exp/eta_th)/esim.gain
    extracted_gains.append(r)

plt.semilogx(brlinspace, extracted_gains, label="Variance/Mean, G=13")


##########Now lets do the flasher

standard_devs_peaks = []
mean_peaks = []
eta_peaks = []

esim_init = TraceSimulation(
    #ampSpec="../data/spe_R11920-RM_ap0.0002.dat",
    timeSpec="../data/bb3_1700v_timing.txt",
    #pulseShape="data/pulse_FlashCam_7dynode_v2a.dat",
    background_rate = 1e9,
    gain=gain,
    no_signal_duration = 1e5,
    noise=1,
)


pulser_init = Pulser(step=esim_init.t_step, duration=esim_init.no_signal_duration, pulse_type="pulsed")
print("pulser std", pulser_init.pulse_std)

I_1 = compute_I1(esim_init.ps_sigma, esim_init.ps_lambda, esim_init.ps_mu, pulser_init)


for j in brlinspace:

    esim = TraceSimulation(
        #ampSpec="../data/spe_R11920-RM_ap0.0002.dat",
        timeSpec="../data/bb3_1700v_timing.txt",
        #pulseShape="data/pulse_FlashCam_7dynode_v2a.dat",
        background_rate = j,
        gain=gain,
        no_signal_duration = 1e5,
        noise=0.8,
    )

    pulse = Pulser(step=esim.t_step,freq=5e6, duration=esim.no_signal_duration, pulse_type="pulsed")
    evts = pulse.generate_all()

    
    evts_br, k_evts = esim.simulateBackground(evts)
    times, pmtSig, uncertainty_pmt = esim.simulatePMTSignal(evts_br, k_evts) #TODO : make uncertainty from the simulatePMTSignal, with ampdist.rvs(). For now sufficient
    eleSig, uncertainty_ele = esim.simulateElectronics(pmtSig, uncertainty_pmt, times)
    stimes, samples, samples_unpro, uncertainty_sampled = esim.simulateADC(times, eleSig, uncertainty_ele, 1)
    dt_resampling = stimes[2]-stimes[1]


    maxs, _ = signal.find_peaks(samples, prominence=10) ###promeminence devrait dependre du gain ? Non
    max_values_stimes = stimes[maxs]
    max_values = samples[maxs]



    ######A good way to do a simple calibration is for example to take the highest peak in the 10 percent of the signal
    cal_index = np.argmax(samples[0:int(len(samples)*0.1)])
    stimes_cal = stimes[cal_index]

    
    plt.figure()
    plt.plot(stimes, samples)

    plt.vlines(stimes_cal,200,600, color="g")

    calibration = stimes_cal


    period = (1.0/pulse.max_frequency)*(1e9) #In ns
    number_pulses = int(pulse.duration // period)

    width_region = period

    maximums_stimes = []
    maximums_values = []

    for i in range(int(0.7*number_pulses)):##Making sure we dont overshoot the signal
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
    plt.xlabel("time in ns" + str(i))
    plt.ylabel("LSB")
    plt.close()
    
    hist, bins = np.histogram(max_values, density=True, bins=30)
    width = (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2

    mean_peaks.append(np.mean(maximums_values))
    standard_devs_peaks.append(np.std(maximums_values))

    eta_peaks.append(np.std(max_values)**2/(np.mean(max_values)-esim.offset-10))##Also removing the prominence



I_2 = compute_I2(esim_init.ps_sigma, esim_init.ps_lambda, esim_init.ps_mu, pulser_init.pulse_std)
theoretical_variance = pulser_init.pe_intensity*gain**2*(esim_init.ampStddev**2+1)*I_2+esim_init.background_rate*gain**2*(esim_init.ampStddev**2+1)*calculate_J2(esim_init.ps_sigma, esim_init.ps_lambda, 0)*1e-9

theoretical_mean = calculate_A(esim_init.ps_sigma, esim_init.ps_lambda, 0)* gain*(pulser_init.pe_intensity*0.027911767902802007)+esim_init.offset+esim_init.singePE_area*esim_init.background_rate*1e-9*gain ###we add the prominence as an offset


extracted_coeff = 1/((pulser_init.pe_intensity*(esim_init.ampStddev**2+1)*I_2
    +esim_init.background_rate*(esim_init.ampStddev**2+1)*calculate_J2(esim_init.ps_sigma, esim_init.ps_lambda, 0)*1e-9)/(calculate_A(esim_init.ps_sigma, esim_init.ps_lambda, 0)*(pulser_init.pe_intensity*0.027911767902802007)+esim_init.singePE_area*esim_init.background_rate*1e-9))


eta = []
eta_th = []

gain_extracted = []

for i in range(len(mean_peaks)):
    eta.append((standard_devs_peaks[i]**2)/(mean_peaks[i]-esim_init.offset))
    #eta_th.append((theoretical_variance)/(theoretical_mean-esim_init.offset))




#plt.plot(gain_linspace, gain_linspace)
plt.semilogx(brlinspace, [x*extracted_coeff/gain for x in eta], ':', label="Pulser, G=13")



plt.semilogx(brlinspace, np.repeat(1, len(brlinspace)), label=r"$\frac{G'}{G}=1$", color="red")



plt.legend(loc="upper left",fontsize=12)


plt.xlabel(r"$f_{NSB}$[Hz]",fontsize=15)
plt.ylabel(r"$\frac{G'}{G}$",fontsize=15)

plt.grid()


plt.show()