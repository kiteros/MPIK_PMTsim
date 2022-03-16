#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm, rv_histogram, randint, poisson, expon, exponnorm
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


class Pulse:

    def execute(esim_init):

        freq = []

        # Start by plotting pulse vs variance
        # Basically the NSB is fixed, and only pulse frequency increases
        # linspace for the pulse frequencz
        
        bl = []

        plt.figure()
        plt.title('Th vs Exp gain')

        # this is a test, should be removed once it works
        for j in np.logspace(5.0, 10.0, num=10):

            freq.append(j)

            pulse = Pulser(step=esim_init.t_step, pulse_type='pulsed', freq=j)
            evts = pulse.generate_all()

            # we need to add random evts that follow a negative exponential for the background rate

            evts_br = esim_init.simulateBackground(evts)

            # pmt signal

            (times, pmtSig) = esim_init.simulatePMTSignal(evts_br)
            eleSig = esim_init.simulateElectronics(pmtSig)

            # adc signal

            (stimes, samples, samples_unpro) = esim_init.simulateADC(times,
                    eleSig)

            (bl_mean, s_mean, std, std_unpro) = esim_init.FPGA(stimes, samples,
                    samples_unpro)
            bl.append(bl_mean)

            plt.plot(stimes, samples, label=j)

        plt.legend(loc='upper left')
        plt.show()

        plt.figure()
        plt.title('Th vs Exp gain')
        plt.plot(freq, bl)
        plt.show()

        return 1
