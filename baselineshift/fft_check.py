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
import csv
import scipy.fftpack
from numpy.fft import fft, ifft
import numpy as np
from scipy import special

#
# configuration
# time analyse = L * (1/Fsample)
#


esim = TraceSimulation(
    ampSpec="../data/spe_R11920-RM_ap0.0002.dat",
    timeSpec="../data/bb3_1700v_timing.txt",
    #pulseShape="data/pulse_FlashCam_7dynode_v2a.dat",
    background_rate = 1e3,
    gain=10,
    no_signal_duration = 1e6,

    ps_mu = 35.11,
    ps_amp = 22.0,
    ps_lambda = 0.0659,
    ps_sigma = 2.7118,
)



def expFourier(lamda, sigma, mu, L):
	#Consider function f(t)=1/(t^2+1)
	#We want to compute the Fourier transform g(w)

	#Discretize time t
	t0=-L
	x = np.linspace(t0, -t0, num=1000)
	Len = len(x) # lenght buffer
	
	dt=x[1]-x[0]
	#Define function
	f=np.exp(-(1/2)*((x-mu)/sigma)**2)

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
	g = np.abs(g[0:Len//2])
	w = w[0:Len//2]

	#Plot Result
	plt.scatter(w,g,color="r")
	theoretical = sigma*np.exp(complex(0,1)*mu*w-(sigma**2*w**2)/2)
	#For comparison we plot the analytical solution
	plt.plot(w,np.abs(theoretical),color="g")

	plt.show()
	plt.close()


	####Works !

	return 0

def gFourier(lamda, sigma, mu,L):
	tau = 1/lamda
	print(tau)
	print(sigma)
	print(mu)
	print(L)
	#Discretize time t

	t0=-65
	x = np.linspace(t0, -t0, num=1000)
	Len = len(x) # lenght buffer
	
	dt=x[1]-x[0]
	#Define function
	f=np.exp((np.sqrt(1/2)*(sigma/tau - (x-mu)/sigma))**2)
	#f=(np.sqrt(1/2)*(sigma/tau - (x-mu)/sigma))**2

	plt.figure()
	plt.plot(x,np.log10(f))
	plt.title("real plot")
	plt.show()


	
	#Compute Fourier transform by numpy's FFT function
	g=fft(f)
	#frequency normalization factor is 2*np.pi/dt
	w = np.fft.fftfreq(f.size)*2*np.pi/dt


	#In order to get a discretisation of the continuous Fourier transform
	#we need to multiply g by a phase factor
	g*=dt*np.exp(-complex(0,1)*w*t0)/(np.sqrt(2*np.pi))
	g = np.abs(g[0:Len//2])
	w = w[0:Len//2]

	#Plot Result
	plt.scatter(w,g,color="r")
	theoretical = -complex(0,1)*sigma*np.exp(  complex(0,1)*mu*w+(sigma**2*w*(2*complex(0,1)+tau*w))/(2*tau) )
	#For comparison we plot the analytical solution
	plt.plot(w,np.abs(theoretical),color="g")

	plt.show()
	plt.close()
	
	return 0


#expFourier(esim.ps_lambda, esim.ps_sigma, esim.ps_mu, esim.pulseShape[0][-1])
gFourier(esim.ps_lambda, esim.ps_sigma, esim.ps_mu, esim.pulseShape[0][-1])


plt.show()