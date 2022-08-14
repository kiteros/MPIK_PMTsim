#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm, rv_histogram, randint, poisson, expon, exponnorm
from scipy.signal import resample
import scipy.integrate as integrate
import scipy
import math 

class Pulser:


    def __init__(
        self, 
        pulser_type="PDL800-D",
        step=0.5,
        duration=1e3,
        pulse_type="none",#"pulsed", "none", "single"
        freq=20e6#Hz
        ):

        self.step = step
        self.duration = duration #ns
        self.depart = 0

        if(pulser_type == "PDL800-D"):
            self.pulse_to_pulse_method = "percent", #percent
            self.pulse_to_pulse_jitter = 0.05#%#2.6ps
            self.max_frequency = freq #Hz (real between 80 MHz and 31.25 KHz)
            self.pulse_width = 20#ns
            self.pulse_std = self.pulse_width/(2*math.sqrt(2*math.log(2)))
            self.average_power = 50e-3 #W
            self.wvl = 600e-9 #wavelenght in m
            self.pe_intensity = 50 #number of PE in a pulse
            self.pulse_type = pulse_type



    
    def generate_peTimes_photon(self, times, pulse):
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

        

        return peTimes

    def generate_peTimes_pe(self, times, pulse,fwmh):

        """
        General the arrival times on the cathode of the PE with respect to the number of PE inside the pulse

        Parameters
        ----------
        times - array_like times of signal in ns

        pulse - Amplitudes of signal

        Returns
        -------
        Array - Starting times of the PE from the incoming signal (on the cathode)
        """

        if self.pulse_type == "none":
            peTimes = []
        elif self.pulse_type == "single":
            #Generate a poissonian variation of the number of PE
            """
            n_pe = poisson.rvs(self.pe_intensity,size=1)[0]
            #generate the pulse from a gaussian + exponential

            bins = np.append(times, times[-1] + times[1] - times[0])
            pulse_dist = rv_histogram((pulse,bins))

            peTimes = pulse_dist.rvs(size=int(n_pe))

            print(n_pe)
            """

            peTimes = np.array([])


            peTimes = np.append(peTimes, np.repeat(10,20))
            peTimes = np.append(peTimes, np.repeat(10000,50))
            peTimes = np.append(peTimes, np.repeat(30000,50))

        elif self.pulse_type == "pulsed":
            #First calculate the number of pulses we can do 

            tsx = np.arange(0, self.duration, 1)

            period = (1.0/self.max_frequency)*(1e9) #In ns
            number_pulses = self.duration // period

            depart = self.depart
            peak_positions = np.array([])

            peTimes = np.array([])
            
            while depart < self.duration:
                #depart stores the positions of the mu of the last pulse
                depart += period

                if self.pulse_to_pulse_method == "absolute":
                    depart += norm.rvs(0.0, self.pulse_to_pulse_jitter*1e-3)
                elif self.pulse_to_pulse_method == "percent":
                    depart += norm.rvs(0.0, period*self.pulse_to_pulse_jitter/(2*math.sqrt(2*math.log(2))))
                peak_positions = np.append(peak_positions, depart)
            
            for i in peak_positions:
                #add a new gaussian for each mu
                curent_signal = norm.pdf(tsx, i, self.pulse_std)
                n_pe = poisson.rvs(self.pe_intensity,size=1)[0] ###poisson dist of the number of pe

                #n_pe = self.pe_intensity

                #bins = np.append(tsx, tsx[-1] + tsx[1] - tsx[0])
                #pulse_dist = rv_histogram((curent_signal,bins))

                
                peTimes = np.append(peTimes, np.array(norm.rvs(loc=i, scale=self.pulse_std, size=int(n_pe))))
                #peTimes = np.append(peTimes, np.repeat(i, n_pe))
                
                
        #print(peTimes)
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

        if self.pulse_type == "single":
            signal = exponnorm.pdf(tsx, K=10, loc=2*fwmh, scale=3)


            """
            plt.figure()
            plt.title("Laser/LED pulse spectrum")
            plt.plot(tsx, signal)
            plt.xlabel("Pulse")
            """

        elif self.pulse_type == "pulsed":

            if depart==0:
                depart = 2*fwmh

            self.depart = depart

            signal = norm.pdf(tsx, depart, self.pulse_std)

        elif self.pulse_type == "pulsed2":

            if depart==0:
                depart = 2*fwmh

            #period in ns
            period = (1.0/self.max_frequency)*(1e9) #In ns

            print(period)

            #generate array for all the mus of the pulses
            peak_positions = np.array([])
            signal = norm.pdf(tsx, depart, fwmh/(2*math.sqrt(2*math.log(2))))


            while depart < self.duration:
                #depart stores the positions of the mu of the last pulse
                depart += period

                if self.pulse_to_pulse_method == "absolute":
                    depart += norm.rvs(0.0, self.pulse_to_pulse_jitter*1e-3)
                elif self.pulse_to_pulse_method == "percent":
                    depart += norm.rvs(0.0, period*self.pulse_to_pulse_jitter/(2*math.sqrt(2*math.log(2))))
                peak_positions = np.append(peak_positions, depart)
            
            for i in peak_positions:
                #add a new gaussian for each mu
                signal += norm.pdf(tsx, i, self.pulse_std)

        
            plt.figure()
            plt.title("Laser/LED pulse spectrum")
            plt.plot(tsx, signal)
            plt.xlabel("Pulse")
            
            

        else:
            signal = np.zeros(len(tsx))

        
        
        return tsx, signal

    def generate_all(self):

        """
        Generate PE start time

        Returns
        -------
        PE start time
        """

        times, pulse = self.generate_pulse(fwmh=self.pulse_width)
        peTimes = self.generate_peTimes_pe(times, pulse,fwmh=self.pulse_width)
        return peTimes

        