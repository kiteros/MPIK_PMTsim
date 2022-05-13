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

def get_enbw(freq, signal):
    df = freq[2]-freq[1]
    enbw_ = 0
    for i in range(len(freq)):
        enbw_+= df*signal[i]**2
    enbw_ = enbw_/(signal[0]**2)
    print("(signal[0])**2",signal[0]**2)
    return enbw_

def integrate(freq, signal):
    ###here it should be signal squared
    df = freq[2]-freq[1]
    inte = 0
    for i in range(len(freq)):
        inte += df*signal[i] 
    return inte

enbw_exp = []
enbw_thexp = []
enbw_th = []
sigmas = []
w_lenght = []

for i in linspace(150, 10000, num=25):

    esim = TraceSimulation(
        ampSpec="data/spe_R11920-RM_ap0.0002.dat",
        timeSpec="data/bb3_1700v_timing.txt",
        #pulseShape="data/pulse_FlashCam_7dynode_v2a.dat",
        background_rate = 3e9,
        gain=10,
        no_signal_duration = 1e6,

        ps_mu = 15.11,
        ps_amp = 1.0,
        ps_lambda = 0.001,#0.0659,
        ps_sigma = 1.0,
        pulse_sampling = 1500,
        pulse_size = int(i)
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

    #plt.figure()
    #plt.plot(x,f)
    #plt.show()

    #Compute Fourier transform by numpy's FFT function
    g=fft(f)
    #frequency normalization factor is 2*np.pi/dt
    w = np.fft.fftfreq(f.size)*2*np.pi/dt


    #In order to get a discretisation of the continuous Fourier transform
    #we need to multiply g by a phase factor
    g*=dt*np.exp(-complex(0,1)*w*t0)/(np.sqrt(2*np.pi))
    g = np.abs(g)
    g = g[0:Len//2]
    w = w[0:Len//2]

    #Plot Result
    #plt.scatter(w,g,color="r")



    ###scaling w
    scaling_w = 0.165
    ###scaling y
    scaling_y = 0.42

    scaled_w = scaling_w*w

    theoretical = scaling_y*A*lamda*np.exp((lamda/2)*(2*mu+lamda*sigma**2))*(1/(lamda +2*np.pi*complex(0,1)*scaled_w))*np.exp(-(lamda+2*np.pi*complex(0,1)*scaled_w)*(mu+lamda*sigma**2))*np.exp((1/2)*sigma**2*(lamda+2*np.pi*complex(0,1)*scaled_w)**2)


    dc_offset = scaling_y*A*lamda*np.exp((lamda/2)*(2*mu+lamda*sigma**2))*(1/lamda)*np.exp(-lamda*(mu+lamda*sigma**2))*np.exp((1/2)*(sigma**2 * lamda**2))

    ##crosscheck okay

    th_abs = scaling_y*A*lamda*np.exp((lamda/2)*(2*mu+lamda*sigma**2)) * ((np.exp(-lamda*(mu+lamda*sigma**2)+(1/2)*sigma**2*(lamda**2-4*np.pi**2*scaled_w**2)))/(np.sqrt(lamda**2+4*np.pi**2*scaled_w**2)))



    th_abs = th_abs[0:Len//2]

    #For comparison we plot the analytical solution
    #plt.plot(w,np.abs(theoretical),color="b")
    #plt.plot(w, th_abs, color="g")
    ########

    #calculer les integrales analytiquement

    #####
    th_enbw = (lamda/4)*(np.exp(sigma**2*lamda**2))*special.erfc(sigma*lamda)

    print("Numerical FFT + numerical ENBW----------------")
    print("ENBW (GHz)", get_enbw(w, g))
    print("Theoretical FFT + Numerical ENBW-------------")
    print("ENBW th (GHz)", get_enbw(w,th_abs))
    print("Theoretical FFT + Theoretical ENBW---------------")
    print("DC offset", dc_offset)
    print("(scaling_y*A)**2",(scaling_y*A)**2)
    print("th enbw", th_enbw)


    #integrals to check
    th_abs_squared = th_abs**2 # notre fonction a check l'integrale
    th_squared = scaling_y**2*(A*lamda)**2 *np.exp(lamda*(2*mu+lamda*sigma**2)) * np.exp(-2*lamda*(mu+lamda*sigma**2)) * np.exp( sigma**2 *lamda**2) *(np.exp(- 4*sigma**2*np.pi**2 *scaled_w**2)/(lamda**2+4*np.pi**2*scaled_w**2))
    th_squared_bis = scaling_y*(A*lamda)**2 *np.exp(lamda*(2*mu+lamda*sigma**2)) * np.exp(-2*lamda*(mu+lamda*sigma**2)) * np.exp( sigma**2 *lamda**2) *(np.exp(- 4*sigma**2*np.pi**2 *scaled_w**2)/(lamda**2+4*np.pi**2*scaled_w**2))
    th_squared = th_squared[0:Len//2]
    th_squared_bis = th_squared_bis[0:Len//2]


    print("----------------------")
    print(integrate(w, th_abs_squared))
    print(integrate(w, th_squared))
    print(integrate(w, th_squared_bis))

    print("----------------------------")

    ######our theoretical value for the integrals
    th_enbw = (lamda/4)*(np.exp(sigma**2*lamda**2))*special.erfc(sigma*lamda)/(scaling_y**2)
    th_integral = (A**2*lamda/4)*(np.exp(sigma**2*lamda**2))*special.erfc(sigma*lamda)
    ##so equal to the scaling**2 function

    print("Numerical FFT + numerical ENBW (GHz)", get_enbw(w, g))
    print("Theoretical FFT + Numerical ENBW (GHz)", get_enbw(w,th_abs))
    print("Theoretical ENBW (GHz):",th_enbw,"=",th_integral,"/",(scaling_y*A)**2)

    enbw_exp.append(get_enbw(w, g))
    enbw_thexp.append(get_enbw(w,th_abs))
    enbw_th.append(th_enbw)

    sigmas.append(i)


plt.figure()
plt.plot(sigmas,enbw_exp,label="Numerical FFT + numerical ENBW (GHz)")
plt.plot(sigmas,enbw_thexp, label="Theoretical FFT + Numerical ENBW (GHz)")
plt.plot(sigmas,enbw_th, label="Theoretical ENBW (GHz)")
plt.xlabel("lenght [ns]")
plt.ylabel("ENBW")
plt.legend(loc="upper right")
plt.show()