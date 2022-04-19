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
import scipy.fftpack

# test function
def fit_function(data, a, b, c, d, e, f, g, h):
    x = data[0]
    y = data[1]
    return a * (x**b) * (y**c) + d*x**e + f*y**g + h

esim = TraceSimulation(
    ampSpec="data/spe_R11920-RM_ap0.0002.dat",
    timeSpec="data/bb3_1700v_timing.txt",
    #pulseShape="data/pulse_FlashCam_7dynode_v2a.dat",
    background_rate = 1e3,
    gain=10,
    no_signal_duration = 1e6,

    ps_mu = 15.11,
    ps_amp = 22.0,
    ps_lambda = 0.0659,
    ps_sigma = 2.7118,
)

def integrateSignal(times, signal):
    """
    Integrates the input signal

    Parameters
    ----------
    times - float
            domain of times
    signal - float
            signal arraz

    Returns
    -------
    sum_
            Integration of the signal
    """
    t_step = times[1]-times[0]

    sum_ = 0
    for i in signal:
        sum_ += i*t_step # maybe wrong
    return sum_

plt.figure()
plt.plot(*esim.pulseShape)



plt.title("Pulse shape")
plt.xlabel("t/ns")
plt.ylabel("A/au")
plt.show()

####fourier transform

T = 1/esim.pulseShape[0][-1]
N = len(esim.pulseShape[0])

yf = scipy.fftpack.fft(esim.pulseShape[1])
xf = np.linspace(0.0, 1.0//(2.0*T), N//2)

fig, ax = plt.subplots()
yf = 2.0/N * np.abs(yf[:N//2])
ax.plot(xf, yf)
plt.show()

#### integrate the transform
inte = integrateSignal(xf, yf)
max_ = max(yf)
print("integrale", inte/max_)

offset_E = 0.24

pulse = Pulser(step=esim.t_step, pulse_type="none")
evts = pulse.generate_all()

sigmas = []
lamdas = []
coeff = []
enbw = []
enbwratio = []
amplitudes = []
averages = []

c_coefficient = []
d_coefficient = []


fig, axs = plt.subplots(2)

for k in np.linspace(1, 50.30, num=3):

	for i in np.linspace(1, 5, num=4):

		for j in np.linspace(0.001, 1, num=4):

			####get the offset from a super low brate

			esim_offset = TraceSimulation(
			    ampSpec="data/spe_R11920-RM_ap0.0002.dat",
			    timeSpec="data/bb3_1700v_timing.txt",
			    #pulseShape="data/pulse_FlashCam_7dynode_v2a.dat",
			    background_rate = 1e3,
			    gain=10,
			    no_signal_duration = 1e4,

			    ps_mu = 15.11,
		        ps_amp = k,
		        ps_lambda = j,
		        ps_sigma = i,
			)

			evts_br, k_evts = esim_offset.simulateBackground(evts)
			times, pmtSig, uncertainty_pmt = esim_offset.simulatePMTSignal(evts_br, k_evts)
			eleSig, uncertainty_ele = esim_offset.simulateElectronics(pmtSig, uncertainty_pmt, times)
			stimes, samples, samples_unpro, uncertainty_sampled = esim_offset.simulateADC(times, eleSig, uncertainty_ele, 1)
			offset, _, _, _, _, _, _, _, _, _ = esim_offset.FPGA(stimes, samples, samples_unpro, uncertainty_sampled, 1, True)

			esim_init = TraceSimulation(
			    ampSpec="data/spe_R11920-RM_ap0.0002.dat",
			    timeSpec="data/bb3_1700v_timing.txt",
			    #pulseShape="data/pulse_FlashCam_7dynode_v2a.dat",
			    background_rate = 3e9,
			    gain=10,
			    no_signal_duration = 1e4,

			    ps_mu = 15.11,
		        ps_amp = k,
		        ps_lambda = j,
		        ps_sigma = i,
			)

			

			evts_br, k_evts = esim_init.simulateBackground(evts)
			times, pmtSig, uncertainty_pmt = esim_init.simulatePMTSignal(evts_br, k_evts)
			eleSig, uncertainty_ele = esim_init.simulateElectronics(pmtSig, uncertainty_pmt, times)
			stimes, samples, samples_unpro, uncertainty_sampled = esim_init.simulateADC(times, eleSig, uncertainty_ele, 1)
			bl_mean, s_mean, std, std_unpro, bl_mean_uncertainty, bl_array, stddev_uncert_mean, stddev_mean, spike, skew = esim_init.FPGA(stimes, samples, samples_unpro, uncertainty_sampled, 1, True)
			ratio = (stddev_mean**2)/(bl_mean-offset)
			ratio = ratio/10
			print(bl_mean, stddev_mean, offset)
			print("ratio", ratio)

			####fourier transform

			T = 1/esim_init.pulseShape[0][-1]
			N = len(esim_init.pulseShape[0])

			yf = scipy.fftpack.fft(esim_init.pulseShape[1])
			xf = np.linspace(0.0, 1.0//(2.0*T), N//2)
			yf = 2.0/N * np.abs(yf[:N//2])

			#### integrate the transform
			inte = integrateSignal(xf, yf)
			max_ = max(yf)
			enbw_ = inte/max_
			print("integrale", enbw_)

			
			lamdas.append(j)

			enbw.append(enbw_)
			coeff.append(ratio)


			
		print("statistics", statistics.fmean(enbw)-offset_E)
		print("ratio", (statistics.fmean(enbw)-offset_E)/statistics.fmean(coeff))
		
		enbwratio.append((statistics.fmean(enbw)-offset_E)/statistics.fmean(coeff))
		axs[0].plot([x - offset_E for x in enbw], coeff, marker='o', label="sigma="+str(i)+"A="+str(k))
		enbw = []
		coeff = []
		sigmas.append(i)

	amplitudes.append(k)
	averages.append(statistics.fmean(enbwratio))
	axs[1].plot(sigmas, [x for x in enbwratio], marker='o', label="A="+str(k))

	### fit

	c, d = np.polyfit(sigmas, enbwratio, 1)
	#axs[1].plot(sigmas, [c*x+d for x in sigmas],'--', label="A="+str(k))
	c_coefficient.append(c)
	d_coefficient.append(d)
	sigmas = []
	enbwratio = []


axs[0].set_xlabel("enbw-offset")
axs[0].set_ylabel("eta coefficient")
axs[1].set_xlabel("sigmas")
axs[1].set_ylabel("(enbw-offset)/eta")
axs[0].legend(loc="upper left")
axs[1].legend(loc="upper left")
plt.show()

#plt.figure()
#plt.scatter(sigmas, enbwratio, marker='o')
#plt.xlabel("sigma value")
#plt.ylabel("enbw/coeff")

#plt.figure()
#plt.plot(amplitudes, coeff)
#plt.xlabel("amplitudes")
#plt.ylabel("enbw")
#plt.legend(loc="upper left")
#plt.show()

#plt.figure()

#((ENBW-offset_E)/eta  - offset_sigma)/sigma en log log et ca devrait etre lineair
#plt.show()

plt.figure()
plt.loglog(amplitudes, [-1*x for x in c_coefficient],marker='o')

a_slope, b_slope = np.polyfit([np.log10(x) for x in amplitudes], [np.log10(-1*x) for x in c_coefficient], 1)
print("a_slope", a_slope)
print("b_slope", b_slope)

plt.xlabel("amplitudes")
plt.ylabel("slope")
plt.show()

plt.figure()
plt.loglog(amplitudes, d_coefficient,marker='o')


###faire le fit sur le offset

a_offset, b_offset = np.polyfit([np.log10(x) for x in amplitudes], [np.log10(x) for x in d_coefficient], 1)
print("a_offset", a_offset)
print("b_offset", b_offset)


plt.xlabel("amplitudes")
plt.ylabel("offset")
plt.show()

plt.figure()


####faire le fit
amplitudes_log = [np.log10(x) for x in amplitudes]
averages_log = [np.log10(x) for x in averages]

a, b = np.polyfit(amplitudes_log, averages_log, 1)
print("a=", a)
print("b=", b)


plt.plot(amplitudes, averages, marker='o')
plt.plot(amplitudes, [x**a * 10**b for x in amplitudes])
plt.xlabel("amplitudes")
plt.ylabel("averages")

plt.figure()
plt.loglog(amplitudes, averages, marker='o')
plt.loglog(amplitudes, [x**a * 10**b for x in amplitudes])
plt.xlabel("amplitudes")
plt.ylabel("averages")




plt.show()

mean_ratio = statistics.fmean(enbwratio)

print("magic ratio : ", mean_ratio)




"""
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

###make the fit
parameters, covariance = curve_fit(fit_function, [sigmas, lamdas], coeff)

model_x_data = np.linspace(min(sigmas), max(sigmas), 30)
model_y_data = np.linspace(min(lamdas), max(lamdas), 30)
# create coordinate arrays for vectorized evaluations
X, Y = np.meshgrid(model_x_data, model_y_data)
# calculate Z coordinate array
Z = fit_function(np.array([X, Y]), *parameters)
# plot surface
ax.plot_surface(X, Y, Z)

for i in range(len(sigmas)):
	ax.scatter(sigmas[i], lamdas[i], coeff[i], marker='o')

ax.set_xlabel("sigma")
ax.set_ylabel("lamda")

print("parameters", parameters)
"""
