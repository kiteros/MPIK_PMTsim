#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm, rv_histogram, randint, poisson, expon, exponnorm, skew
from scipy.signal import resample
from pulser import Pulser
import statistics
import scipy.integrate as integrate
import math 
import sys
import scipy.special as sse
from scipy.optimize import curve_fit
import scipy.special as sse

np.set_printoptions(threshold=sys.maxsize)

pulseShape="data/pulse_FlashCam_7dynode_v2a.dat"
ps = np.loadtxt(pulseShape, unpack=True)


def fit_func(x, A, l, s, m):
    return A*0.5*l*np.exp(0.5*l*(2*m+l*s*s-2*x))*sse.erfc((m+l*s*s-x)/(np.sqrt(2)*s)) # exponential gaussian


max_value = ps[0][-1]
print(max_value)
print("------")
x = np.linspace(0, 100, num=1138)
x1000 = np.linspace(0, 1000, num=1138)
popt, pcov = curve_fit(fit_func, x, ps[1])
#print(ps[0])

plt.figure()
plt.plot(ps[0], ps[1], label="data")
new_mu = (popt[3]/100)*max_value
new_lamda = popt[1]/(max_value/100)
new_amplitude = popt[0]*(max_value/100)
new_sigma = popt[2]*(max_value/100)
plt.plot(x, fit_func(x, new_amplitude, new_lamda, new_sigma, new_mu), label="fit")
K = 1/(new_lamda*new_sigma)
plt.plot(x, new_amplitude*exponnorm.pdf(x, K, loc=new_mu, scale=new_sigma), label="exponnorm")
plt.legend(loc="upper right")
plt.show()

