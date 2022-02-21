#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm, rv_histogram, randint, poisson, expon
from scipy.signal import resample
import scipy.integrate as integrate
import scipy
import math 

class Pulser:


    def __init__(
        self, 
        pulser_type="PDL800-D",
        step=0.5,
        duration=400.0):

        self.step = step
        self.duration = duration #ns

        if(pulser_type == "PDL800-D"):
            self.pulse_to_pulse_jitter = 2.6 #ps
            self.max_frequency = 20e6 #Hz (real between 80 MHz and 31.25 KHz)
            self.pulse_width = 20#ns
            self.average_power = 50e-3 #W
            self.wvl = 600e-9 #wavelenght in m

    
    def generate_peTimes(self, times, pulse):
        """
        General the arrival times on the cathode of the PE with respect to the number of photons present in the pulse

        Parameters
        ----------
        times - array_like times of signal in ns

        pulse - Amplitudes of signal

        Returns
        -------
        Array - Starting times of the PE from the incoming signal (on the cathode)
        """

        energy = integrate.cumtrapz(self.average_power*pulse, times*(10e-9), initial=0)[-1]
        normalization = integrate.cumtrapz(pulse, times*(10e-9), initial=0)[-1]

        reduction_factor = 1e8 #As not to simulate too much PE 

        h = 6.62607004e-22 #kg.ps^-1
        c = 3.0e-4 #speed of light in m.ps^-1
        pht_energy = h * c / self.wvl
        nb_photon = energy / pht_energy #number of photons in the pulse

        bins = np.append(times, times[-1] + times[1] - times[0])
        pulse_dist = rv_histogram((pulse/normalization, bins))

        peTimes = pulse_dist.rvs(size=int(nb_photon/reduction_factor))

        plt.figure()
        plt.title("Laser/LED pulse spectrum")
        plt.plot(times, pulse)
        plt.xlabel("Pulse")

        return peTimes
    

    def generate_pulse(self, depart=0, fwmh=3.0):
        """
        Generate a LED/Laser pulse 

        Parameters
        ----------
        depart - First mu of the pulse in ns

        fwmh - Full width at mid heigh in ns

        Returns
        -------
        Tuple - times and signal
        """

        tsx = np.arange(0, self.duration, 1)

        if depart==0:
            depart = 2*fwmh

        #period in ns
        period = (1.0/self.max_frequency)*(1e9)
        print(period)

        #generate array for all the mus of the pulses
        peak_positions = np.array([])
        signal = norm.pdf(tsx, depart, fwmh/(2*math.sqrt(2*math.sqrt(2))))


        while depart < self.duration:
            #depart stores the positions of the mu of the last pulse
            depart += period
            depart += norm.rvs(0.0, self.pulse_to_pulse_jitter)
            peak_positions = np.append(peak_positions, depart)
        
        for i in peak_positions:
            #add a new gaussian for each mu
            signal += norm.pdf(tsx, i, fwmh/(2*math.sqrt(2*math.sqrt(2))))
        
        return tsx, signal

    def generate_all(self):

        """
        Generate PE start time

        Returns
        -------
        PE start time
        """

        times, pulse = self.generate_pulse(fwmh=self.pulse_width)
        peTimes = self.generate_peTimes(times, pulse)
        return peTimes

        