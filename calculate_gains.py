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

class GainCalculator:


    def __init__(
        self, 
        gain_min=2,
        gain_max=15,

        n_train_1=7,
        n_train_2=7,

        n_exp_1=7,
        n_exp_2=7,

        bk_min=5.0,
        bk_max=7.0, #Both in log scale
        nsb_var = 0.1,
        ):

        self.gain_min = gain_min
        self.gain_max = gain_max
        self.n_train_1 = n_train_1
        self.n_train_2 = n_train_2
        self.n_exp_1 = n_exp_1
        self.n_exp_2 = n_exp_2
        self.bk_min = bk_min
        self.bk_max = bk_max
        self.nsb_var = nsb_var


    def extract_gain(self):


        #ampSpec = np.loadtxt("data/bb3_1700v_spe.txt", unpack=True)

        #Amplitude spectrum obtained from spe_R11920-RM_ap0.0002.dat
        ampSpec = np.loadtxt("data/bb3_1700v_spe.txt", unpack=True)
        timeSpec = "data/bb3_1700v_timing.txt"
        pulseShape = np.loadtxt("data/pulse_FlashCam_7dynode_v2a.dat", unpack=True)

        # init class
        esim_init = TraceSimulation(
            ampSpec="data/spe_R11920-RM_ap0.0002.dat",
            timeSpec="data/bb3_1700v_timing.txt",
            pulseShape="data/pulse_FlashCam_7dynode_v2a.dat",
        )

        if esim_init.show_graph:
            plt.figure()
            plt.title("Amplitude spectrum")
            plt.plot(*esim_init.ampSpec)
            plt.xlabel("Amplitude")

            plt.figure()
            plt.title("Time spectrum")
            plt.plot(*esim_init.timeSpec)
            plt.xlabel("Time/ns")

            plt.figure()
            plt.title("Pulse shape")
            plt.plot(*esim_init.pulseShape)
            plt.xlabel("Time/ns")
            plt.ylabel("Amplitude")

        #Create a pulser class
        

        if esim_init.gain_extraction_method == "baseline":

            pulse = Pulser(step=esim_init.t_step, pulse_type="none")

            #This will obv generate no events
            evts = pulse.generate_all()

            #print(evts)

            #Function mostly to plot the gains
            
            #We dont care about the slope of this one
            gains, slope, slopes_uncertainties = self.loop_gain_bl(evts=evts, gain_min=self.gain_min, gain_max=self.gain_max, 
                    n1=self.n_train_1, n2=self.n_train_2, bk_min=self.bk_min, bk_max=self.bk_max, nsb_var=self.nsb_var)
            

            plt.figure()
            plt.title("Th vs Exp gain")
            plt.plot(gains, gains)
            

            #We want to do the same thing with different parameters, for a first we will modifiy the background rate variation

            #ADC noise variation
            for i in np.linspace(1.0,1.0, num=5):

                gains, slopes, slopes_uncertainties = self.loop_gain_bl(evts=evts, gain_min=self.gain_min, gain_max=self.gain_max, 
                    n1=self.n_train_1, n2=self.n_train_2, bk_min=self.bk_min, bk_max=self.bk_max, nsb_var=self.nsb_var, noise_=i)
                coeff, coeff_uncertainty = self.calculate_coeff(evts=evts)
                exp_gain = []
                exp_gain_unc = []

                for i, q in enumerate(slopes):
                    unc_s = slopes_uncertainties[i]

                    exp_gain.append(q/coeff)
                    exp_gain_unc.append(unc_s/coeff + (q/(coeff*coeff))*coeff_uncertainty)


                #plt.plot(gains, exp_gain, label=i)
                plt.errorbar(gains, exp_gain, yerr=exp_gain_unc, label=i)

            plt.legend(loc="upper left")

            plt.xlabel("gain")
            plt.ylabel("gain")  
            plt.show()

        elif esim_init.gain_extraction_method == "pulse":
            #Start by plotting pulse vs variance
            #Basically the NSB is fixed, and only pulse frequency increases

            #linspace for the pulse frequencz

            freq = []
            bl = []

            plt.figure()
            plt.title("Th vs Exp gain")
            
            
            #this is a test, should be removed once it works
            for j in np.logspace(5.0, 10.0, num=10):
                freq.append(j)
                


                pulse = Pulser(step=esim_init.t_step, pulse_type="pulsed", freq=j)
                evts = pulse.generate_all()

                

                #we need to add random evts that follow a negative exponential for the background rate
                evts_br = esim_init.simulateBackground(evts)

                # pmt signal
                times, pmtSig = esim_init.simulatePMTSignal(evts_br)
                eleSig = esim_init.simulateElectronics(pmtSig)

                # adc signal
                stimes, samples, samples_unpro = esim_init.simulateADC(times, eleSig)

                bl_mean, s_mean, std, std_unpro = esim_init.FPGA(stimes, samples, samples_unpro)
                bl.append(bl_mean)

                plt.plot(stimes, samples, label=j)

            plt.legend(loc="upper left")
            plt.show()

            plt.figure()
            plt.title("Th vs Exp gain")
            plt.plot(freq, bl)
            plt.show()



        return 1

    def line_1(self, x, a, b):
        return a * x + b

    def line_(self, p, x):
        return p[0] * x + p[1]

    def loop_gain_pulse(self):
        return 1

    def calculate_coeff(self, evts):
        #first extract the slopes of from the different gains
        gains, slopes, slopes_uncertainties = self.loop_gain_bl(evts=evts, gain_min=self.gain_min, gain_max=self.gain_max, 
                    n1=self.n_train_1, n2=self.n_train_2, bk_min=self.bk_min, bk_max=self.bk_max, nsb_var=self.nsb_var)
        #Then calculate the coefficient
        #coeff = np.polyfit(gains, slopes, 1)[0]


        ####Here we have an error on coeff

        popt, pcov = curve_fit(self.line_1, gains, slopes, sigma=slopes_uncertainties)


        coeff = popt[0]
        coeff_uncertainty = pcov[0,0]**0.5



        return coeff, coeff_uncertainty

    

    def loop_gain_bl(self, evts, gain_min, gain_max, n1, n2, bk_min, bk_max, nsb_var, noise_=1.0):

        #print(evts)

        slopes = []
        slopes_uncertainties = []
        gains = []
        #Varying the gain
        for j in np.linspace(gain_min, gain_max, num=n1):

            bl_mean_array = []
            s_mean_array = []
            freq = []
            std_dev = []
            std_dev_unpro = []
            theoretical = []
            ratio_bl_exp = []
            bl_mean_uncer_array = []

            gains.append(j)


            #Varying the background rate
            for i in np.logspace(bk_min, bk_max, num=n2):
                #print("background",i)
                esim = TraceSimulation(
                    ampSpec="data/spe_R11920-RM_ap0.0002.dat",
                    timeSpec="data/bb3_1700v_timing.txt",
                    pulseShape="data/pulse_FlashCam_7dynode_v2a.dat",
                    background_rate = i,
                    gain=j,
                    max_nsb_var=nsb_var,
                    noise=noise_,
                )

                #we need to add random evts that follow a negative exponential for the background rate

                evts_br, k_evts = esim.simulateBackground(evts)

                # pmt signal
                times, pmtSig, uncertainty = esim.simulatePMTSignal(evts_br, k_evts) #TODO : make uncertainty from the simulatePMTSignal, with ampdist.rvs(). For now sufficient


                eleSig, uncertainty = esim.simulateElectronics(pmtSig, uncertainty)

                # adc signal
                stimes, samples, samples_unpro, uncertainty_sampled = esim.simulateADC(times, eleSig, uncertainty)

                bl_mean, s_mean, std, std_unpro, bl_mean_uncertainty = esim.FPGA(stimes, samples, samples_unpro, uncertainty_sampled)

                bl_mean_array.append(bl_mean)
                bl_mean_uncer_array.append(bl_mean_uncertainty)

                s_mean_array.append(s_mean)
                freq.append(i)
                std_dev.append(std*std)
                std_dev_unpro.append(std_unpro)
                theoretical.append(esim.singePE_area*esim.gain*esim.background_rate* 1e-9 + esim.offset)
                ratio_bl_exp.append(bl_mean/(esim.singePE_area*esim.gain*esim.background_rate* 1e-9 + esim.offset))

                #gain_ = esim.extractGain(stimes, samples, samples_unpro, bl_mean)
                
                if 0:
                    plt.figure()
                    plt.title("Simulated ADC output")
                    plt.plot(stimes + esim.plotOffset, samples)
                    plt.plot(stimes + esim.plotOffset, np.ones(len(stimes))*bl_mean)
                    plt.xlabel("Time/ns")
                    plt.ylabel("ADC output/LSB")

                    plt.figure()
                    plt.title("Simulated signal")
                    #plt.scatter(evts_br, np.zeros(evts_br.shape))
                    #plt.bar(times, pmtSig)
                    plt.plot(stimes + esim.plotOffset, samples_unpro)
                    plt.xlabel("Time/ns")
                    plt.ylabel("Amplitude")



            #Lets calculate the uncertainty on offset :
            #Uncertainty on the fit

            #print("bl mean",bl_mean_uncer_array)

            popt, pcov = curve_fit(self.line_1, freq, bl_mean_array, sigma=bl_mean_uncer_array)
            #print("a =", popt[0], "+/-", pcov[0,0]**0.5)
            #print("b =", popt[1], "+/-", pcov[1,1]**0.5)


            offset = popt[1]
            offset_uncertainty = pcov[1,1]**0.5

            #Substraction of uncertainty
            quad_model = odr.Model(self.line_)


            ####Need to find a way to get the uncertainty of stddev
            ############It is not very good but let's assume it is the same as blmean

            data = odr.RealData(bl_mean_array - offset, std_dev, sx=bl_mean_uncer_array + offset_uncertainty, sy=bl_mean_uncer_array + offset_uncertainty)
            odr_ = odr.ODR(data, quad_model, beta0=[1., 0.])

            out = odr_.run()

            popt2 = out.beta
            perr2 = out.sd_beta

            
            #popt2, pcov2 = curve_fit(self.line_,  bl_mean_array - offset, std_dev, sigma=bl_mean_uncer_array + offset_uncertainty)
            slope = popt2[0]
            slope_error = perr2[0]

            print(slope, slope_error)
            #print( "slope", slope)#



            slopes.append(slope)
            slopes_uncertainties.append(slope_error)

        return gains, slopes, slopes_uncertainties
