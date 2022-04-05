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
import pandas as pd

def func(x, a, b, c, d, e):

    return e*np.array(x)**2 + d*np.array(x)**3 + a*np.array(x)**2 + b* np.array(x) + c

def func_exp(x, a, b, c):

    return a*np.array(x)**2 + b* np.array(x) + c

csvdat = pd.read_csv('table_traces.csv')                                       


current_gain = -1
gains_list = []
all_background_rates = []
all_ratios = []

background_rates = []
ratios = []

datas = []
data = []

gain_plusratio = []

for ch in csvdat.iterrows():

	gain_plusratio.append([ch[1]['True gain'], ch[1]['Baseline/Var'], ch[1]['True NSB Background']])

	if ch[1]['True gain'] != current_gain:

		#print(ch[1]['True gain'])
		gains_list.append(ch[1]['True gain'])
		current_gain = ch[1]['True gain']

		all_background_rates.append(background_rates)
		all_ratios.append(ratios)
		datas.append(data)

		background_rates = []
		ratios = []
		data = []

		background_rates.append(ch[1]['True NSB Background'])
		ratios.append(ch[1]['Baseline/Var'])

		
		#data.append([ch[1]['Baseline mean'], ch[1]['Signal std']**2/ch[1]['Baseline mean']])

	else:

		background_rates.append(ch[1]['True NSB Background'])
		ratios.append(ch[1]['Baseline/Var'])

	data.append([ch[1]['True NSB Background'], ch[1]['Baseline/Var'], ch[1]["Baseline mean - offset"], ch[1]["Signal Variance"]])



all_background_rates.append(background_rates)
all_ratios.append(ratios)
datas.append(data)

all_background_rates.pop(0)
all_ratios.pop(0)
datas.pop(0)

background_rates = []
ratios = []

fig, axs = plt.subplots(3)
plt.figure()


for j in range(len(all_ratios)):
	new_data = datas[j]
	new_data.sort(key=lambda x:x[0])

	new_data = [item for item in new_data if not math.isinf(item[1])]

	print(new_data)

	set_x = [x[0] for x in new_data]
	set_y = [x[1] for x in new_data]
	baseline = [x[2] for x in new_data]
	var = [x[3] for x in new_data]

	set_x_log = [np.log10(x[0]) for x in new_data]
	set_y_log = [np.log10(x[1]) for x in new_data]


	moving_average = []
	span = 40

	###ets do a rolling average
	for i in range(len(set_y)):
		if i <span:
			average = set_y[i]
			moving_average.append(average)
		else :
			average = (set_y[i]+set_y[i-1]+set_y[i-2]+set_y[i-3]+set_y[i-4])/4
			moving_average.append(average)


	set_x = np.array(set_x)
	set_y = np.array(set_y)
	set_x_log = np.array(set_x_log)
	set_y_log = np.array(set_y_log)


	axs[0].loglog(set_x, set_y, label="Gain="+str(format(gains_list[j],".2f")))
	axs[1].loglog(set_x, baseline, label="Gain="+str(format(gains_list[j],".2f")))
	axs[2].loglog(set_x, var, label="Gain="+str(format(gains_list[j],".2f")))
	#axs.loglog(set_x, moving_average)

	#poly = np.polyfit(set_x_log, set_y_log, deg=1)
	
	#popt, pcov = curve_fit(func, set_x_log, set_y_log)

	##axs.loglog(10**set_x_log, 10**func(set_x_log, *popt), 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f, %5.3f, %5.3f' % tuple(popt))

	#axs.plot(np.polyval(poly, set_x_log), '--', label='fit'+str(format(gains_list[j],".2f")))

	###poly fit plus print the fit

axs[0].set_xlabel("Background rate [Hz]")
axs[1].set_xlabel("Background rate [Hz]")
axs[2].set_xlabel("Background rate [Hz]")
axs[0].set_ylabel("variance/baseline mean []")
axs[1].set_ylabel("baseline []")
axs[2].set_ylabel("variance []")
axs[0].legend(loc="upper right")
axs[1].legend(loc="upper right")
axs[2].legend(loc="upper right")
axs[0].grid()
axs[1].grid()
axs[2].grid()

#axs.title.set_text("Gain estimation with 8 gain sample x 8 backgorund rate sample")


"""

plt.figure()
plt.scatter([x[0] for x in gain_plusratio], [x[1] for x in gain_plusratio])

annotation = [str(x[2]) for x in gain_plusratio]

for i, label in enumerate(annotation):
	plt.annotate(label, (gain_plusratio[i][0], gain_plusratio[i][1]))

plt.xlabel("True gain")
plt.ylabel("Var/Baseline")

"""

plt.show()

##########


