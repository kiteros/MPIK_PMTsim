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
	print("0th signal", signal[0])
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

def expnm(lamda, sigma, mu, A, x):
	return (A*lamda/2)*np.exp((lamda/2)*(2*mu+lamda*sigma**2-2*x))*special.erfc((mu+lamda*sigma**2-x)/(np.sqrt(2)*sigma))

def numerical_DC_Offset(lamda, sigma, mu, A, L):
	x = np.linspace(0, L, num=1000)
	f = expnm(lamda, sigma, mu, A, x)

	result = integrate.quad(lambda x: expnm(lamda, sigma, mu, A, x), 0, L)/L
	
	return result

def theoretical_DC_Offset(lamda, sigma, mu, A, L):
	f = (A/(2*L))*(special.erf((L-mu)/(np.sqrt(2)*sigma)) + special.erf(mu/(np.sqrt(2)*sigma)) 
		+ np.exp(0.5*lamda*(2*mu+lamda*sigma**2))*special.erfc((mu+lamda*sigma**2)/(np.sqrt(2)*sigma))
		- np.exp(0.5*lamda*(-2*L+2*mu+lamda*sigma**2)) * special.erfc((-L+mu+lamda*sigma**2)/(np.sqrt(2)*sigma)))
	#f = (A/(2*L))*special.erfc(L/(np.sqrt(2)*sigma))+np.exp((lamda**2*sigma**2)/2)*special.erfc((lamda*sigma)/np.sqrt(2))-np.exp((1/2)*lamda*(-2*L+lamda*sigma**2))*special.erfc((-L+lamda*sigma**2)/(np.sqrt(2)*sigma))
	return f

def fourier_transform(lamda, sigma, mu, A, L):

	x = np.linspace(0, L, num=1000)
	f_exp = np.exp(0.5*lamda*(2*mu+lamda*sigma**2-2*x))
	Len = len(x) # lenght buffer
	Tsample = x[2]-x[1]
	yf = fft(f_exp)
	yf = 1.0/Len * np.abs(yf[0:Len//2])
	xf = np.linspace(0.0, 1.0/(2.0*Tsample), Len//2)


	### theoretical

	th_transform = np.exp(0.5*l*(2*mu+lamda*sigma**2))

	return 0

def expFourier(lamda, sigma, mu, L):
	#Consider function f(t)=1/(t^2+1)
	#We want to compute the Fourier transform g(w)

	#Discretize time t
	t0=-L
	x = np.linspace(t0, -t0, num=1000)
	xpos = np.linspace(0, -t0, num=500)
	Len = len(x) # lenght buffer
	LenPos = len(xpos)
	
	dt=x[1]-x[0]
	dt_pos = xpos[1]-xpos[0]
	#Define function
	f=np.exp((lamda/2)*(2*mu+lamda*sigma**2-2*np.abs(x)))
	f_notabs=np.exp((lamda/2)*(2*mu+lamda*sigma**2-2*xpos))

	#Compute Fourier transform by numpy's FFT function
	g=fft(f)
	g_notabs=fft(f_notabs)
	#frequency normalization factor is 2*np.pi/dt
	w = np.fft.fftfreq(f.size)*2*np.pi/dt
	w_pos = np.fft.fftfreq(f_notabs.size)*2*np.pi/dt_pos


	#In order to get a discretisation of the continuous Fourier transform
	#we need to multiply g by a phase factor
	g*=dt*np.exp(-complex(0,1)*w*t0)/(np.sqrt(2*np.pi))
	g_notabs *= dt_pos*np.exp(complex(0,1)*w_pos*t0)/(np.sqrt(2*np.pi))
	g = g[0:Len//2]
	w = w[0:Len//2]
	w_pos = w_pos[0:LenPos//2]
	g_notabs = g_notabs[0:LenPos//2]

	#Plot Result
	plt.scatter(w,g,color="r")
	plt.scatter(w_pos, np.abs(g_notabs), color="g")
	#For comparison we plot the analytical solution
	plt.plot(w,(np.exp((lamda/2)*(2*mu+lamda*sigma**2))*lamda*np.sqrt(2/np.pi))/(lamda**2+w**2),color="g")

	plt.show()
	plt.close()


	####Works !

	return 0

def erfFourier(lamda, sigma, mu, L):
	
	x = np.linspace(-L,L, num=1000)
	f = special.erfc((mu+lamda*sigma**2-np.abs(x))/(np.sqrt(2)*sigma))


	Len = len(x) # lenght buffer
	
	dt=x[1]-x[0]
	#Define function

	#Compute Fourier transform by numpy's FFT function
	g=fft(f)
	#frequency normalization factor is 2*np.pi/dt
	w = np.fft.fftfreq(f.size)*2*np.pi/dt


	#In order to get a discretisation of the continuous Fourier transform
	#we need to multiply g by a phase factor
	g*=dt*np.exp(complex(0,1)*w*L)/(np.sqrt(2*np.pi))
	g = np.abs(g)
	

	#Plot Result
	plt.plot(w,g,color="r")
	#For comparison we plot the analytical solution
	analytic = np.exp(2 *np.pi * complex(0,1) * w * (mu+lamda*sigma**2)/(np.sqrt(2)*sigma))*(complex(0,1)*np.exp(-0.5*w**2*sigma**2)*np.sqrt(2/np.pi))/w
	plt.plot(w,np.abs(analytic),color="g")

	plt.show()
	plt.close()
	


	return 0

def convolution(lamda, sigma, mu, A, L, t, x2):
	x = np.linspace(0,L, num=1000)
	dt=x[1]-x[0]
	w = np.fft.fftfreq(x.size)*2*np.pi/dt

	w = w[len(x)//2:] +w[:len(x)//2]

	f = np.abs( (np.exp((lamda/2)*(2*mu+lamda*sigma**2))*lamda*np.sqrt(2/np.pi))/(lamda**2+w**2))


	plt.figure()
	plt.plot(w, f)


	g = np.abs( np.exp(2 *np.pi * complex(0,1) * w * (mu+lamda*sigma**2)/(np.sqrt(2)*sigma))*(complex(0,1)*np.exp(-0.5*w**2*sigma**2)*np.sqrt(2/np.pi))/w )
	g[g > 1e308] = 20


	plt.figure()
	plt.plot(w, g)

	conv = np.abs(np.convolve(f, g, 'same')) * (A*lamda/2)

	#take only the positive half
	w = w[len(w)//2:]
	conv = conv[len(conv)//2:]

	plt.figure()
	plt.plot(w, conv)




	"""
	#Discretize time t
	t0=-t[-1]
	time = np.linspace(t0, -t0, num=1000)
	Len = len(time) # lenght buffer
	
	dt=time[1]-time[0]

	#Compute Fourier transform by numpy's FFT function
	g2=fft(x2)
	#frequency normalization factor is 2*np.pi/dt
	w2 = np.fft.fftfreq(g2.size)*2*np.pi/dt


	#In order to get a discretisation of the continuous Fourier transform
	#we need to multiply g by a phase factor
	g2*=dt*np.exp(-complex(0,1)*w2*t0)/(np.sqrt(2*np.pi))
	g2 = np.abs(g2[0:Len//2])
	w2 = w2[0:Len//2]

	print(w2)

	print("0rth frequency", g2[0])

	#Plot Result
	plt.scatter(w2,g2,color="r")
	"""

	L = len(t) # lenght buffer
	Tsample = t[2]-t[1]
	yf = fft(x2)
	yf = 1.0/L * np.abs(yf[0:L//2])
	xf = np.linspace(0.0, 1.0/(2.0*Tsample), L//2)


	plt.plot(xf, yf)
	return 0

def example(lamda, sigma, mu, L):
	
	return 0


def crosschecks():
	#verify that the 0rth frequency is indeed the integral 
	x = np.linspace(-10, 10, num=1000)
	f = np.exp(-(x/np.sqrt(2)))/sqrt(2*np.pi)

	t0 = -10
	Len = len(x) # lenght buffer
	
	dt=x[1]-x[0]

	#Compute Fourier transform by numpy's FFT function
	g=fft(f)
	#frequency normalization factor is 2*np.pi/dt
	w = np.fft.fftfreq(g.size)*2*np.pi/dt


	#In order to get a discretisation of the continuous Fourier transform
	#we need to multiply g by a phase factor
	g*=dt*np.exp(-complex(0,1)*w*t0)/(np.sqrt(2*np.pi))
	g = np.abs(g)
	#w = w[0:Len//2]

	plt.figure()
	plt.plot(w, g)

	L = len(x) # lenght buffer
	Tsample = x[2]-x[1]
	yf = fft(f)
	yf = 1.0/L * np.abs(yf[0:L//2])
	xf = np.linspace(0.0, 1.0/(2.0*Tsample), L//2)


	plt.plot(xf, yf)

	return 0


enbw, xf, yf = ENBW(esim.pulseShape[0], esim.pulseShape[1])

print("ENBW",enbw)
print("DC offset np.mean()", np.mean(esim.pulseShape[1]))
print("Lenght of pulse", esim.pulseShape[0][-1])
print("theoretical DC offset", theoretical_DC_Offset(esim.ps_lambda, esim.ps_sigma, esim.ps_mu, esim.ps_amp, esim.pulseShape[0][-1]))

print("numericalDC offset", numerical_DC_Offset(esim.ps_lambda, esim.ps_sigma, esim.ps_mu, esim.ps_amp, esim.pulseShape[0][-1]))

print("integrate signal", integrateSignal(esim.pulseShape[0], esim.pulseShape[1]))
print("(integrate signal)/L", integrateSignal(esim.pulseShape[0], esim.pulseShape[1])/esim.pulseShape[0][-1])

###############Compare the fourier transforms

#fourier_transform(esim.ps_lambda, esim.ps_sigma, esim.ps_mu, esim.ps_amp, esim.pulseShape[0][-1])


#example(esim.ps_lambda, esim.ps_sigma, esim.ps_mu, esim.pulseShape[0][-1])


expFourier(esim.ps_lambda, esim.ps_sigma, esim.ps_mu, esim.pulseShape[0][-1])
#erfFourier(esim.ps_lambda, esim.ps_sigma, esim.ps_mu, esim.pulseShape[0][-1])

#convolution(esim.ps_lambda, esim.ps_sigma, esim.ps_mu, esim.ps_amp,esim.pulseShape[0][-1], esim.pulseShape[0], esim.pulseShape[1])

#crosschecks()

plt.show()