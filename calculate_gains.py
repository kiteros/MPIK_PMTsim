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

class GainCalculator:


    def __init__(
        self, 
        gain_min=2,
        gain_max=15,

        n_train_1=3,#minimum 3 point for covariance
        n_train_2=3,

        n_exp_1=3,
        n_exp_2=3,

        bk_min=6.0,
        bk_max=8.0, #Both in log scale
        nsb_var = 0.1,
        ):

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

        """
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
        """

        self.esim_init = esim_init

        self.gain_min = gain_min
        self.gain_max = gain_max
        self.n_train_1 = n_train_1
        self.n_train_2 = n_train_2
        self.n_exp_1 = n_exp_1
        self.n_exp_2 = n_exp_2
        self.bk_min = bk_min
        self.bk_max = bk_max
        self.nsb_var = nsb_var

        ##n_exp not used so far

        self.nb_lines = 1

        print("estimated time", (self.nb_lines*self.n_train_1*self.n_train_2/12)%60, "minutes")


    def line_1(self, x, a, b):
        return a * x + b

    def line_(self, p, x):
        return p[0] * x + p[1]

    def loop_gain_pulse(self):
        return 1


    def extract_gain(self):


        #ampSpec = np.loadtxt("data/bb3_1700v_spe.txt", unpack=True)

        

        #Create a pulser class
        

        if self.esim_init.gain_extraction_method == "baseline":

            pulse = Pulser(step=self.esim_init.t_step, pulse_type="none")

            #This will obv generate no events
            evts = pulse.generate_all()
            
            #We dont care about the slope of this one
            #gains, slope, slopes_uncertainties = self.loop_gain_bl(evts=evts, gain_min=self.gain_min, gain_max=self.gain_max, 
                    #n1=self.n_train_1, n2=self.n_train_2, bk_min=self.bk_min, bk_max=self.bk_max, nsb_var=self.nsb_var)
            

            gains = np.arange(self.gain_min, self.gain_max+1, 1)

            fig, ax = plt.subplots()
            ax.set_title("Th vs Exp gain")
            ax.plot(gains, gains)
            

            #We want to do the same thing with different parameters, for a first we will modifiy the background rate variation



            #ADC noise variation

            gain_2 = []
            noises = []
            exp_gain_unc = []

            for i in np.linspace(0.5,0.5, num=self.nb_lines):
                #print(i)
                noises.append(i)

                #training
                coeff, coeff_uncertainty, offset_coeff = self.calculate_coeff(evts=evts, noi=i)

                #experience
                gains, slopes, slopes_uncertainties = self.loop_gain_bl(evts=evts, gain_min=self.gain_min, gain_max=self.gain_max, 
                    n1=self.n_exp_1, n2=self.n_exp_2, bk_min=self.bk_min, bk_max=self.bk_max, nsb_var=self.nsb_var, noise_=i)

                #gains, slopes, slopes_uncertainties = self.loop_gain_bl(evts=evts, gain_min=2, gain_max=2, 
                    #n1=self.n_exp_1, n2=self.n_exp_2, bk_min=self.bk_min, bk_max=self.bk_max, nsb_var=self.nsb_var, noise_=i)

                exp_gain = []
                exp_gain_unc = []

                for i, q in enumerate(slopes):
                    unc_s = slopes_uncertainties[i]

                    exp_gain.append(q/coeff)
                    exp_gain_unc.append(unc_s/coeff + (q/(coeff*coeff))*coeff_uncertainty)


                ax.plot(gains, exp_gain-offset_coeff, label='exp')
                #plt.errorbar(gains, exp_gain, yerr=exp_gain_unc, label=i)
                #print(exp_gain)
                

                #gain_2.append(exp_gain[0])
                
            ax.fill_between(gains, [a - b for a, b in zip(gains, exp_gain_unc)], [a + b for a, b in zip(gains, exp_gain_unc)], alpha=0.2)
            ax.legend(loc="upper left")

            #plt.figure()
            #plt.plot(noises, gain_2)


            plt.show()
            
            

        elif self.esim_init.gain_extraction_method == "pulse":
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

        elif self.esim_init.gain_extraction_method == "debug":

            pulse = Pulser(step=self.esim_init.t_step, pulse_type="none")

            #This will obv generate no events
            evts = pulse.generate_all()

            plt.figure()
            plt.title("Simulated ADC output")

            for i in range(5):
                #Repeat and generate n different signals with same uncertainties

                esim = TraceSimulation(
                    ampSpec="data/spe_R11920-RM_ap0.0002.dat",
                    timeSpec="data/bb3_1700v_timing.txt",
                    pulseShape="data/pulse_FlashCam_7dynode_v2a.dat",
                )

                #we need to add random evts that follow a negative exponential for the background rate

                evts_br, k_evts = esim.simulateBackground(evts)

                # pmt signal
                times, pmtSig, uncertainty_pmt = esim.simulatePMTSignal(evts_br, k_evts) #TODO : make uncertainty from the simulatePMTSignal, with ampdist.rvs(). For now sufficient


                eleSig, uncertainty_ele = esim.simulateElectronics(pmtSig, uncertainty_pmt)

                # adc signal
                stimes, samples, samples_unpro, uncertainty_sampled = esim.simulateADC(times, eleSig, uncertainty_ele)

                bl_mean, s_mean, std, std_unpro, bl_mean_uncertainty, bl_array,stddev_uncert_mean = esim.FPGA(stimes, samples, samples_unpro, uncertainty_sampled)

                th_bg = np.ones(len(stimes))*(esim.singePE_area*esim.gain*esim.background_rate* 1e-9 + esim.offset)
                
                plt.plot(stimes + esim.plotOffset, samples, label=i)
                plt.plot(stimes + esim.plotOffset, np.ones(len(stimes))*bl_mean)
                #plt.plot(stimes + esim.plotOffset, bl_array)
                plt.plot(stimes + esim.plotOffset, th_bg)
                plt.fill_between(stimes + esim.plotOffset, [a - b for a, b in zip(th_bg, uncertainty_sampled)], [a + b for a, b in zip(th_bg, uncertainty_sampled)], alpha=0.2)
                
                
                """
                plt.figure()
                plt.title("Electronic signal")
                #plt.scatter(evts_br, np.zeros(evts_br.shape))
                #plt.bar(times, pmtSig)
                plt.plot(times, eleSig, label=i)
                plt.fill_between(times, [a - b for a, b in zip(eleSig, uncertainty_ele)], [a + b for a, b in zip(eleSig, uncertainty_ele)], alpha=0.2)
                plt.xlabel("Time/ns")
                plt.ylabel("Amplitude")
                plt.legend(loc="upper left")

                plt.figure()
                plt.title("PMT signal")
                #plt.scatter(evts_br, np.zeros(evts_br.shape))
                #plt.bar(times, pmtSig)
                plt.plot(times, pmtSig,label=i)
                plt.fill_between(times, [a - b for a, b in zip(pmtSig, uncertainty_pmt)], [a + b for a, b in zip(pmtSig, uncertainty_pmt)], alpha=0.2)
                plt.xlabel("Time/ns")
                plt.ylabel("Amplitude")
                plt.legend(loc="upper left")
                """
            plt.xlabel("Time/ns")
            plt.ylabel("ADC output/LSB")
            plt.legend(loc="upper left")
            plt.show()

        elif self.esim_init.gain_extraction_method == "under_c":

            pulse = Pulser(step=self.esim_init.t_step, pulse_type="none")

            #This will obv generate no events
            evts = pulse.generate_all()

            plt.figure()
            plt.title("bl underestimate")

            means_ = []
            brate = []

            for j in np.logspace(3.0, 9.0, num=5):

                mean_ = []
                brate.append(math.log10(j))

                for i in range(2):
                    #Repeat and generate n different signals with same uncertainties
                    print(i,j)

                    esim = TraceSimulation(
                        ampSpec="data/spe_R11920-RM_ap0.0002.dat",
                        timeSpec="data/bb3_1700v_timing.txt",
                        pulseShape="data/pulse_FlashCam_7dynode_v2a.dat",
                        background_rate = j,
                    )

                    #we need to add random evts that follow a negative exponential for the background rate

                    evts_br, k_evts = esim.simulateBackground(evts)

                    # pmt signal
                    times, pmtSig, uncertainty_pmt = esim.simulatePMTSignal(evts_br, k_evts) #TODO : make uncertainty from the simulatePMTSignal, with ampdist.rvs(). For now sufficient


                    eleSig, uncertainty_ele = esim.simulateElectronics(pmtSig, uncertainty_pmt)

                    # adc signal
                    stimes, samples, samples_unpro, uncertainty_sampled = esim.simulateADC(times, eleSig, uncertainty_ele)

                    bl_mean, s_mean, std, std_unpro, bl_mean_uncertainty, bl_array, stddev_uncert_mean = esim.FPGA(stimes, samples, samples_unpro, uncertainty_sampled)

                    th_bg = np.ones(len(stimes))*(esim.singePE_area*esim.gain*esim.background_rate* 1e-9 + esim.offset)

                    mean_.append(bl_mean)

                means_.append(esim.singePE_area*esim.gain*esim.background_rate* 1e-9 + esim.offset - statistics.fmean(mean_))


            z = np.polyfit(brate, means_, 1)
            slope_ = z[0]
            offset_ = z[1]
            print(slope_, offset_)

            plt.plot(brate, [x * slope_ + offset_ for x in brate])
            plt.plot(brate, means_)
            plt.show()
                    
        elif self.esim_init.gain_extraction_method == "blstddev":

            bl_stddev = []
            uncertainty_stddev = []
            br_rate = []
            stddev_mean_e = []

            pulse = Pulser(step=self.esim_init.t_step, pulse_type="none")

            #This will obv generate no events
            evts = pulse.generate_all()

            for j in np.logspace(3.0, 9, num=5):

                #br_rate.append(j)
                br_rate.append(math.log10(j))
                print(j)

                esim = TraceSimulation(
                    ampSpec="data/spe_R11920-RM_ap0.0002.dat",
                    timeSpec="data/bb3_1700v_timing.txt",
                    pulseShape="data/pulse_FlashCam_7dynode_v2a.dat",
                    background_rate = j,
                )

                #we need to add random evts that follow a negative exponential for the background rate

                evts_br, k_evts = esim.simulateBackground(evts)

                # pmt signal
                times, pmtSig, uncertainty_pmt = esim.simulatePMTSignal(evts_br, k_evts) #TODO : make uncertainty from the simulatePMTSignal, with ampdist.rvs(). For now sufficient


                eleSig, uncertainty_ele = esim.simulateElectronics(pmtSig, uncertainty_pmt)

                # adc signal
                stimes, samples, samples_unpro, uncertainty_sampled = esim.simulateADC(times, eleSig, uncertainty_ele)

                bl_mean, s_mean, std, std_unpro, bl_mean_uncertainty, bl_array, stddev_uncert_mean, stddev_mean = esim.FPGA(stimes, samples, samples_unpro, uncertainty_sampled)

                th_bg = np.ones(len(stimes))*(esim.singePE_area*esim.gain*esim.background_rate* 1e-9 + esim.offset)

                bl_stddev.append(math.log10(np.std(bl_array)))
                #bl_stddev.append(np.std(bl_array))
                uncertainty_stddev.append(math.log10(stddev_uncert_mean))
                #uncertainty_stddev.append(stddev_uncert_mean)
                stddev_mean_e.append(math.log10(stddev_mean))

            plt.figure()
            plt.plot(br_rate, bl_stddev, label="bl")
            plt.plot(br_rate, uncertainty_stddev, label="uncertainty")
            plt.plot(br_rate, stddev_mean_e, label="stddev_mean")
            plt.legend(loc="upper left")
            plt.show()

            #Lets fit an exponential


        return 1

    

    def calculate_coeff(self, evts, noi):
        #first extract the slopes of from the different gains
        gains, slopes, slopes_uncertainties = self.loop_gain_bl(evts=evts, gain_min=self.gain_min, gain_max=self.gain_max, 
                    n1=self.n_train_1, n2=self.n_train_2, bk_min=self.bk_min, bk_max=self.bk_max, nsb_var=self.nsb_var, noise_=noi)


        #Make sure we have train for the coeff (the bigger, the more precise)


        #Then calculate the coefficient
        #coeff = np.polyfit(gains, slopes, 1)[0]

        plt.figure()
        plt.plot(gains, slopes, 'bo')
        plt.fill_between(gains, [a - b for a, b in zip(slopes, slopes_uncertainties)], [a + b for a, b in zip(slopes, slopes_uncertainties)], alpha=0.2)



        ####Here we have an error on coeff

        popt, pcov = curve_fit(self.line_1, gains, slopes, sigma=slopes_uncertainties)


        coeff = popt[0]
        offset_coeff = popt[1]
        coeff_uncertainty = pcov[0,0]**0.5

        plt.plot(gains, [x * coeff + offset_coeff for x in gains])



        return coeff, coeff_uncertainty, offset_coeff

    

    def loop_gain_bl(self, evts, gain_min, gain_max, n1, n2, bk_min, bk_max, nsb_var, noise_=1.0):

        #print(evts)

        slopes = []
        slopes_uncertainties = []
        gains = []
        #Varying the gain
        plt.figure()
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
            for i in np.linspace(10**bk_min, 10**bk_max, num=n2):
                print("background",i)
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
                times, pmtSig, uncertainty_pmt = esim.simulatePMTSignal(evts_br, k_evts) #TODO : make uncertainty from the simulatePMTSignal, with ampdist.rvs(). For now sufficient


                eleSig, uncertainty_ele = esim.simulateElectronics(pmtSig, uncertainty_pmt)

                # adc signal
                stimes, samples, samples_unpro, uncertainty_sampled = esim.simulateADC(times, eleSig, uncertainty_ele)

                bl_mean, s_mean, std, std_unpro, bl_mean_uncertainty, bl_array, stddev_uncert_mean = esim.FPGA(stimes, samples, samples_unpro, uncertainty_sampled)

                bl_mean_array.append(bl_mean)
                bl_mean_uncer_array.append(bl_mean_uncertainty)

                s_mean_array.append(s_mean)
                freq.append(i)
                std_dev.append(std*std)
                std_dev_unpro.append(std_unpro)
                theoretical.append(esim.singePE_area*esim.gain*esim.background_rate* 1e-9 + esim.offset)
                ratio_bl_exp.append(bl_mean/(esim.singePE_area*esim.gain*esim.background_rate* 1e-9 + esim.offset))

                #gain_ = esim.extractGain(stimes, samples, samples_unpro, bl_mean)
                
                if esim.show_signal_graphs:
                    plt.figure()
                    plt.title("Simulated ADC output")
                    plt.plot(stimes + esim.plotOffset, samples, label=i)
                    plt.plot(stimes + esim.plotOffset, np.ones(len(stimes))*bl_mean)
                    plt.fill_between(stimes + esim.plotOffset, [a - b for a, b in zip(samples, uncertainty_sampled)], [a + b for a, b in zip(samples, uncertainty_sampled)], alpha=0.2)
                    plt.xlabel("Time/ns")
                    plt.ylabel("ADC output/LSB")
                    plt.legend(loc="upper left")

                    
                    plt.figure()
                    plt.title("Electronic signal")
                    #plt.scatter(evts_br, np.zeros(evts_br.shape))
                    #plt.bar(times, pmtSig)
                    plt.plot(times, eleSig, label=i)
                    plt.fill_between(times, [a - b for a, b in zip(eleSig, uncertainty_ele)], [a + b for a, b in zip(eleSig, uncertainty_ele)], alpha=0.2)
                    plt.xlabel("Time/ns")
                    plt.ylabel("Amplitude")
                    plt.legend(loc="upper left")

                    plt.figure()
                    plt.title("PMT signal")
                    #plt.scatter(evts_br, np.zeros(evts_br.shape))
                    #plt.bar(times, pmtSig)
                    plt.plot(times, pmtSig,label=i)
                    plt.fill_between(times, [a - b for a, b in zip(pmtSig, uncertainty_pmt)], [a + b for a, b in zip(pmtSig, uncertainty_pmt)], alpha=0.2)
                    plt.xlabel("Time/ns")
                    plt.ylabel("Amplitude")
                    plt.legend(loc="upper left")



            #Lets calculate the uncertainty on offset :
            #Uncertainty on the fit

            #print("bl mean",bl_mean_uncer_array)

            popt, pcov = curve_fit(self.line_1, freq, bl_mean_array, sigma=bl_mean_uncer_array)
            #print("a =", popt[0], "+/-", pcov[0,0]**0.5)
            #print("b =", popt[1], "+/-", pcov[1,1]**0.5)


            offset = popt[1]
            offset_uncertainty = pcov[1,1]**0.5

            

            #Try to reimplement the classical curve fit with the self variable
            if self.esim_init.slope_method == "odr":

                #Substraction of uncertainty
                quad_model = odr.Model(self.line_)


                ####Need to find a way to get the uncertainty of stddev
                ############It is not very good but let's assume it is the same as blmean

                data = odr.RealData(bl_mean_array - offset, std_dev, sx=bl_mean_uncer_array + offset_uncertainty, sy=bl_mean_uncer_array + offset_uncertainty)
                #data = odr.RealData(bl_mean_array - offset, std_dev, sx=np.repeat(1,len(bl_mean_uncer_array + offset_uncertainty)), sy=np.repeat(1,len(bl_mean_uncer_array + offset_uncertainty)))
                odr_ = odr.ODR(data, quad_model, beta0=[1., 0.])

                out = odr_.run()

                popt2 = out.beta
                perr2 = out.sd_beta
                slope = popt2[0]
                slope_error = perr2[0]

                #print(slope, slope_error)

            elif self.esim_init.slope_method == "classical":
                popt2, pcov2 = curve_fit(self.line_1,  bl_mean_array - offset, std_dev, sigma=bl_mean_uncer_array + offset_uncertainty)
                slope = popt2[0]

                slope_error = pcov2[0,0]**0.5

                #print(slope, slope_error)


            ##print bl_mean-offset vs stddev with slopes as a reference

            if self.esim_init.show_graph == True:

                #print("bl_mean", bl_mean_uncer_array)
                #print("offset", offset_uncertainty)

                plt.errorbar(bl_mean_array-offset, std_dev, xerr=bl_mean_uncer_array + offset_uncertainty, yerr=bl_mean_uncer_array + offset_uncertainty, fmt='o', label="data")
                #plt.errorbar(bl_mean_array-offset, std_dev, xerr=np.repeat(1,len(bl_mean_uncer_array + offset_uncertainty)), yerr=np.repeat(1,len(bl_mean_uncer_array + offset_uncertainty)), fmt='o', label="data")
                plt.plot(bl_mean_array-offset, [x * slope for x in bl_mean_array-offset], label="gain="+str(j))
                
                

            slopes.append(slope)
            slopes_uncertainties.append(slope_error)

        plt.xlabel("bl_mean_array-offset")
        plt.ylabel("std_dev")
        plt.legend(loc="upper left")
        return gains, slopes, slopes_uncertainties
