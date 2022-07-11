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
import pandas as pd

def func(x, a, b, c, d, e):

    return e*np.array(x)**2 + d*np.array(x)**3 + a*np.array(x)**2 + b* np.array(x) + c

def func_exp(x, a, b, c):

    return a*np.array(x)**2 + b* np.array(x) + c

plt.figure()

####for now std gain is only on one point, but we can check for every point that it
#corresponds to the theoretical standard deviation

std_gains = []
th_std = 0

for filename in os.listdir('../tables/'):
	f = os.path.join('../tables/', filename)

	csvdat = pd.read_csv(f)                                       


	current_gain = -1
	gains_list = []
	all_background_rates = []
	all_ratios = []

	background_rates = []
	ratios = []

	datas = []
	data = []

	gain_plusratio = []

	eta = 0

	for ch in csvdat.iterrows():

		eta = ch[1]['eta']

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

		data.append([ch[1]['True NSB Background'], ch[1]['Baseline/Var'], ch[1]["Baseline mean - offset"], ch[1]["Signal Variance"], ch[1]["lamda poisson"], ch[1]["amp std"], ch[1]["ADC noise"], ch[1]["pulse shape total sum"], ch[1]["N samples"]])



	all_background_rates.append(background_rates)
	all_ratios.append(ratios)
	datas.append(data)

	all_background_rates.pop(0)
	all_ratios.pop(0)
	datas.pop(0)

	background_rates = []
	ratios = []

	#fig, axs = plt.subplots(1)


	
	
	for j in range(len(all_ratios)):
		new_data = datas[j]
		new_data.sort(key=lambda x:x[0]) #array of NSB sorted

		new_data = [item for item in new_data if not math.isinf(item[1])]


		min_percent = 0.0
		max_percent = 1.0

		set_x = [x[0] for x in new_data][int(min_percent*len(new_data)):int(max_percent*len(new_data))]
		set_y = [x[1] for x in new_data][int(min_percent*len(new_data)):int(max_percent*len(new_data))]
		baseline = [x[2] for x in new_data][int(min_percent*len(new_data)):int(max_percent*len(new_data))]
		var = [x[3] for x in new_data][int(min_percent*len(new_data)):int(max_percent*len(new_data))]
		lamda_poisson = [x[4] for x in new_data][int(min_percent*len(new_data)):int(max_percent*len(new_data))]
		ampStddev = [x[5] for x in new_data][int(min_percent*len(new_data)):int(max_percent*len(new_data))]
		noise = [x[6] for x in new_data][int(min_percent*len(new_data)):int(max_percent*len(new_data))]
		pulseshape_totalsum = [x[7] for x in new_data][int(min_percent*len(new_data)):int(max_percent*len(new_data))]
		N_samples = [x[8] for x in new_data][int(min_percent*len(new_data)):int(max_percent*len(new_data))]

		set_x_log = [np.log10(x[0]) for x in new_data][int(min_percent*len(new_data)):int(max_percent*len(new_data))]
		set_y_log = [np.log10(x[1]) for x in new_data][int(min_percent*len(new_data)):int(max_percent*len(new_data))]


		moving_average = []
		span = 40

		color = np.random.rand(3,)
		tr_color = np.append(color,0.3)

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

		total_std = []


		


		for k in range(len(lamda_poisson)):

			N_samples[k] = 20000
			print(N_samples[k])

			tot = np.sqrt((ampStddev[k]**2*(lamda_poisson[k]+lamda_poisson[k]**2)+lamda_poisson[k])*pulseshape_totalsum[k])
			tot = tot*gains_list[j]+noise[k]
			tot = 2*tot**2/(eta*sqrt(N_samples[k])*baseline[k])

			
			total_std.append(tot)
			print("##############")
			print(gains_list[j])
			print(lamda_poisson[k])
			print(noise[k])
			print(baseline[k])
			print(tot)
		


		#plt.semilogx(set_x, np.repeat(eta, len(set_x)), color='red', label="Theoretical eta")
		plt.fill_between(set_x, np.subtract(set_y/eta,total_std), np.add(set_y/eta,total_std), color=tr_color)
			
		
		plt.semilogx(set_x, set_y/gains_list[j], label="Gain="+str(format(gains_list[j],".2f")), color=color)

		###executed every line in case only one gain
		std_gains.append(set_y[5]/eta)#fifth element
		th_std = total_std[5]
		print(th_std)

		
		

		#axs[1].loglog(set_x, baseline, label="Gain="+str(format(gains_list[j],".2f")))
		

		#axs[2].loglog(set_x, var, label="Gain="+str(format(gains_list[j],".2f")))
		#axs.loglog(set_x, moving_average)

		#poly = np.polyfit(set_x_log, set_y_log, deg=1)
		
		#popt, pcov = curve_fit(func, set_x_log, set_y_log)

		##axs.loglog(10**set_x_log, 10**func(set_x_log, *popt), 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f, %5.3f, %5.3f' % tuple(popt))

		#axs.plot(np.polyval(poly, set_x_log), '--', label='fit'+str(format(gains_list[j],".2f")))

		###poly fit plus print the fit

#axs[1].plot(set_x, np.repeat(eta, len(set_x)), color='red', label="Theoretical eta")
#for i in gains_list:

print("standard exp",np.std(std_gains))
print("standard th", th_std)

plt.xlabel("Background rate [Hz]")
#axs[1].set_xlabel("Background rate [Hz]")
plt.ylabel("Variance/(BaselineShift*G)")
#axs[1].set_ylabel("variance/baseline mean []")
plt.legend(loc="upper right")
#axs[1].legend(loc="upper left")
plt.grid()
#axs[1].grid()

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
