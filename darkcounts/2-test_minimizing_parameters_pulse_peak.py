#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import rv_histogram, randint, poisson, expon, exponnorm
from scipy.signal import resample
import scipy.integrate as integrate

import sys

sys.path.insert(0, '/home/jebach/Documents/flashcam/pmt-trace-simulation-master/PMTtraceSIM_draft/simulation')
sys.path.insert(0, '/home/jebach/Documents/flashcam/pmt-trace-simulation-master/PMTtraceSIM_draft/debug_fcts')
sys.path.insert(0, '/home/jebach/Documents/flashcam/pmt-trace-simulation-master/PMTtraceSIM_draft/baselineshift')
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

"""
This file tests the list of parameters provided and looks at the standard deviation of the relative gain
"""


### input_ = [N,G,Br]
### output_ = [P,W,R,K]

input_ = [0.8, 7, 316227]
output_ = [3.689655172413793, 1, 3, 3.43103448275862]
#[3.931034482758621, 0, 1, 2.5206896551724136]
#[4.625, 0, 4, 3.0999999999999996]
#[3.75, 1, 3, 4.0]
#[3.5714285714285716, 1, 4, 3.0]

window_ = ''

if output_[1] == 0:
	window_ = 'tukey'
elif output_[1] == 1:
	window_ = 'blackman'


ph = PeakHistogram(
	noise=input_[0],
	background_rate=input_[2],
	gain_linspace=[input_[1]],
	graphs=False,
	verbose=False,
	prominence = output_[0],
    window = window_,
    resampling_rate=output_[2],
    kde_banwidth=output_[3],
    trace_lenght=2e5,

)


print("###############STATE#############")
print("Noise", input_[0])
print("Gain", input_[1])
print("background rate", input_[2])
print("##############PARAMETERS###############")
print("Prominence", output_[0])
print("window", window_)
print("Resampling rate", output_[2])
print("kde kde_banwidth", output_[3])
relative = ph.get_relative_gain_array()[0]
print("relative gain", relative)
print("Extracted gain", relative*input_[1])





brlinspace = np.logspace(5,6,num=5)

gainlinspace = np.linspace(5,10,num=5)

noise_linspace = [0.8, 1.5, 2]


fig, ax = plt.subplots(1, 3,constrained_layout=True)

iter_ = 0

for noise in noise_linspace:
	
	for gain in gainlinspace:
		extracted_gains = []
		for background_rate_ in brlinspace:
			ph = PeakHistogram(
				noise=noise,
				background_rate=background_rate_,
				gain_linspace=[gain],
				graphs=False,
				verbose=False,
				prominence = output_[0],
			    window = window_,
			    resampling_rate=output_[2],
			    kde_banwidth=output_[3],
			    trace_lenght=2e5,

			)

			r = ph.get_relative_gain_array()[0]
			extracted_gains.append(r)

		ax[iter_].semilogx(brlinspace, extracted_gains, label="gain="+str(gain))
	ax[iter_].set_title("Noise = "+str(noise))

	ax[iter_].legend(loc="upper left")
	ax[iter_].set_xlabel("NSB rate")
	ax[iter_].set_ylabel("G'/G")

	iter_ += 1

plt.show()