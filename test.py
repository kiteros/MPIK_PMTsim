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
from numpy.fft import fft, ifft
import numpy as np

#
# configuration
# time analyse = L * (1/Fsample)
#


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
    step = times[2]-times[1]

    sum_ = 0
    for i in signal:
        sum_ += i*step # maybe wrong
    return sum_

def get_enbw(freq, signal):
	df = freq[2]-freq[1]
	enbw_ = 0
	for i in range(len(freq)):
		enbw_+= df*signal[i]**2
	enbw_ = enbw_/(signal[0]**2)
	return enbw_

def get_enbw_lin(freq, signal):
	df = freq[2]-freq[1]
	enbw_ = 0
	for i in range(len(freq)):
		enbw_+= df*signal[i]
	enbw_ = enbw_/(signal[0])
	return enbw_


def ENBW(t, x):

	L = len(t) # lenght buffer
	Tsample = t[2]-t[1]
	yf = fft(x)
	yf = 1.0/L * np.abs(yf[0:L//2])
	xf = np.linspace(0.0, 1.0/(2.0*Tsample), L//2)
	enbw = get_enbw(xf, yf)
	return enbw, xf, yf

enbw, xf, yf = ENBW(esim.pulseShape[0], esim.pulseShape[1])

print("ENBW",enbw)
print("DC offset", np.mean(esim.pulseShape[1]))

print("integrate signal", integrateSignal(esim.pulseShape[0], esim.pulseShape[1]))
print("mean", integrateSignal(esim.pulseShape[0], esim.pulseShape[1])/esim.pulseShape[0][-1])