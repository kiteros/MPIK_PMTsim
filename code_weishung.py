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
from scipy import special

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

L=esim.pulseShape[0][-1]
mu = esim.ps_mu
sigma = esim.ps_sigma
lamda = esim.ps_lambda
A = esim.ps_amp


t0=-L
x = np.linspace(t0, -t0, num=1000)
Len = len(x) # lenght buffer

dt=x[1]-x[0]
#Define function
f=(A*lamda/2)*np.exp((lamda/2)*(2*mu+lamda*sigma**2-2*x))*special.erfc((mu+lamda*sigma**2-x)/(np.sqrt(2)*sigma))

plt.figure()
plt.plot(x,f)
plt.show()

#Compute Fourier transform by numpy's FFT function
g=fft(f)
#frequency normalization factor is 2*np.pi/dt
w = np.fft.fftfreq(f.size)*2*np.pi/dt


#In order to get a discretisation of the continuous Fourier transform
#we need to multiply g by a phase factor
g*=dt*np.exp(-complex(0,1)*w*t0)/(np.sqrt(2*np.pi))
g = np.abs(g)


#Plot Result
plt.scatter(w,g,color="r")

theoretical = A*lamda*np.exp((lamda/2)*(2*mu+lamda*sigma**2))*(1/(lamda +2*np.pi*complex(0,1)*w))*np.exp(-(lamda+2*np.pi*complex(0,1)*w)*(mu+lamda*sigma**2))*np.exp((1/2)*sigma**2*(lamda+2*np.pi*complex(0,1)*w)**2)


#For comparison we plot the analytical solution
plt.plot(w,np.abs(theoretical),color="b")


plt.show()
plt.close()