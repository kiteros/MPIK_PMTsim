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


brate = []
gain = []

blmean = []
std_bl = []
stddev = []
spikes = []
skews = []
Y = []
X = []

#Amplitude spectrum obtained from spe_R11920-RM_ap0.0002.dat
ampSpec = np.loadtxt("data/bb3_1700v_spe.txt", unpack=True)
timeSpec = "data/bb3_1700v_timing.txt"
pulseShape = np.loadtxt("data/pulse_FlashCam_7dynode_v2a.dat", unpack=True)

# init class
esim_init = TraceSimulation(
    ampSpec="data/spe_R11920-RM_ap0.0002.dat",
    timeSpec="data/bb3_1700v_timing.txt",
    pulseShape="data/pulse_FlashCam_7dynode_v2a.dat",
)


for filename in os.listdir('exports/'):
    f = os.path.join('exports/', filename)
    # checking if it is a file
    if os.path.isfile(f):
        #Just add to an array

        #Check if it only line one

        print(filename.split("line=")[1].split(".")[0])

        if filename.split("line=")[1].split(".")[0] == str(1):
        	B = float(filename.split("=")[1].split(";")[0])
        	G = float(filename.split("=")[2].split(";")[0])
        	brate.append(B)
        	gain.append(G)
        	Y.append([B, G])

        	print(f)

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

	            

	            #_, _, _, _, _, _, _, _, spike, skew = esim_init.FPGA(stimes, samples, samples_unpro, uncertainty_sampled, 1, False)
	            blmean.append(bl_mean)
	            std_bl.append(std)
	            stddev.append(stddev_mean)
	            spikes.append(spike)
	            skews.append(skew)
	            X.append([bl_mean, std, stddev_mean, spike])


###We might only want those that are not in double


plt.figure()

color_x = []
color_y = []


for i in range(len(brate)):
	maximum = max(brate)
	maximum_gain = max(gain)

	color_x.append(brate[i]/maximum)
	color_y.append(gain[i]/maximum_gain)

	plt.scatter(brate[i], gain[i], color = [(brate[i]/maximum,gain[i]/maximum_gain,0.5)])


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(len(blmean)):
	ax.scatter(blmean[i], stddev[i], spikes[i], c=[(color_x[i], color_y[i], 0.5)], marker='o')




plt.show()

regr = linear_model.LinearRegression()
regr.fit(X, Y)


##################generate a trace to then compare
predict_brate = 1e8
predict_gain = 15


pulse = Pulser(step=esim_init.t_step, pulse_type="none")
evts = pulse.generate_all()

#generate it
esim = TraceSimulation(
    ampSpec="data/spe_R11920-RM_ap0.0002.dat",
    timeSpec="data/bb3_1700v_timing.txt",
    pulseShape="data/pulse_FlashCam_7dynode_v2a.dat",
    background_rate = predict_brate,
    gain=predict_gain,
)

evts_br, k_evts = esim.simulateBackground(evts)

# pmt signal
times, pmtSig, uncertainty_pmt = esim.simulatePMTSignal(evts_br, k_evts) #TODO : make uncertainty from the simulatePMTSignal, with ampdist.rvs(). For now sufficient


eleSig, uncertainty_ele = esim.simulateElectronics(pmtSig, uncertainty_pmt, times)

# adc signal
stimes, samples, samples_unpro, uncertainty_sampled = esim.simulateADC(times, eleSig, uncertainty_ele, 1)

#This part should be done all the time, even when it is loaded
bl_mean, _, std, _, _, _, _, stddev_mean, spike, skew = esim.FPGA(stimes, samples, samples_unpro, uncertainty_sampled, 1,  False)


predict = regr.predict([[bl_mean, std, stddev_mean, spike]])

print("----------predict")
print(predict)
print("background rate", predict[0][0])
print("gain", predict[0][1])
print("----------real")
print("background_rate", predict_brate)

print("gain", predict_gain)