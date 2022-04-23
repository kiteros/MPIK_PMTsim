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
from numpy.fft import fft, ifft

##

esim = TraceSimulation(
    ampSpec="data/spe_R11920-RM_ap0.0002.dat",
    timeSpec="data/bb3_1700v_timing.txt",
    #pulseShape="data/pulse_FlashCam_7dynode_v2a.dat",
    background_rate = 1e3,
    gain=10,
    no_signal_duration = 1e6,

    ps_mu = 15.11,
    ps_amp = 1.0,
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


###### test

t0 = 0
t1 = 2
n_samples = 10000

xs = np.linspace(t0, t1, n_samples)
ys = 7 * np.sin(15 * 2 * np.pi * xs) + 3 * np.sin(13 * 2 * np.pi * xs) + 100*np.sqrt(xs)

plt.subplot(2, 1, 1)
plt.plot(xs, ys)

print(integrateSignal(xs, ys))

np_fft = np.fft.fft(ys)
amplitudes = 2 / n_samples * np.abs(np_fft) 
frequencies = np.fft.fftfreq(n_samples) * n_samples * 1 / (t1 - t0)

plt.subplot(2, 1, 2)
plt.semilogx(frequencies[:len(frequencies) // 2], amplitudes[:len(np_fft) // 2])

plt.show()

######################################


# sampling rate
dt = esim.pulseShape[0][2]-esim.pulseShape[0][1]
T = esim.pulseShape[0][-1]
sr = 1/dt
n_samples = len(esim.pulseShape[0])
t = np.linspace(min(esim.pulseShape[0]), max(esim.pulseShape[1]), num=n_samples)
x = esim.pulseShape[1]



plt.figure(figsize = (8, 6))
plt.plot(t, x, 'r')
plt.ylabel('Amplitude')

plt.show()

X = fft(x)
X_real = np.real(X)
X_imag = np.imag(X)

freq = np.fft.fftfreq(n_samples) * n_samples * 1/T
amplitudes = 2/n_samples * np.abs(X)

one_side = len(x)//2

print(len(x))

plt.figure(figsize = (12, 6))
plt.subplot(121)

plt.plot(freq[:one_side],amplitudes[:one_side], 'b')

plt.xlabel('Freq (Hz)')
plt.ylabel('FFT Amplitude |X(freq)|')
plt.xlim(0, 10)

plt.subplot(122)
plt.plot(t, ifft(X), 'r')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()