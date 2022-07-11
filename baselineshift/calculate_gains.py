#!/usr/bin/env python3

import numpy as np
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
from scipy.interpolate import interp1d


class GainCalculator:


    def __init__(
        self, 
        gain_min=2,
        gain_max=15,

        n_train_1=3,#minimum 3 point for covariance ####gains ########keep 12 gains, new standard
        n_train_2=12,

        ###12, 600, rerun the simulation next time i can

        n_exp_1=3,
        n_exp_2=12,

        bk_min=2.0,
        bk_max=10, #Both in log scale, limited to 9 for now
        nsb_var = 0.0,
        noise = 1.0,

        load_files = True, 
        ):

        #Amplitude spectrum obtained from spe_R11920-RM_ap0.0002.dat
        ampSpec = np.loadtxt("../data/bb3_1700v_spe.txt", unpack=True)
        timeSpec = "../data/bb3_1700v_timing.txt"
        pulseShape = np.loadtxt("../data/pulse_FlashCam_7dynode_v2a.dat", unpack=True)

        # init class
        esim_init = TraceSimulation(
            ampSpec="../data/spe_R11920-RM_ap0.0002.dat",
            timeSpec="../data/bb3_1700v_timing.txt",
            pulseShape="../data/pulse_FlashCam_7dynode_v2a.dat",
        )

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
        self.load_files = load_files
        self.noise = noise

        self.coefficient = 0
        self.coeff_uncertainty = 0
        self.offsets = []

        self.nb_lines = 1

        print("estimated time", (self.nb_lines*self.n_train_1*self.n_train_2/12)%60, "minutes")

        plt.figure()
        plt.plot(*esim_init.pulseShape)
        plt.title("Pulse shape")
        plt.xlabel("t/ns")
        plt.ylabel("A/au")
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.plot(*esim_init.timeSpec)
        plt.title("Time spectrum")
        plt.xlabel("t/ns")
        plt.ylabel("Probability")
        plt.subplot(1, 2, 2)
        plt.plot(*esim_init.ampSpec)
        plt.title("Amplitude spectrum")
        plt.xlabel("A/au")
        plt.ylabel("Probability")
        plt.show()





    def line_1(self, x, a, b):
        #Line function for fit
        return a * x + b

    def line_(self, p, x):
        return p[0] * x + p[1]

    def loop_gain_pulse(self):
        return 1

    def gaussian(self, x, sigma, mu):
        fct = (1/(sigma * math.sqrt(2*math.pi)))*math.exp(-(1/2)*(x-mu)**2/sigma**2)
        return fct


    def train(self):
        pulse = Pulser(step=self.esim_init.t_step, pulse_type="none")
        evts = pulse.generate_all()

        self.coefficient, self.coeff_uncertainty, offset_coeff, self.offsets = self.calculate_coeff(evts=evts, noi=self.noise, line_nb_=1)
        print(self.offsets)
        if self.esim_init.verbose:
            print("coefficient", self.coefficient)

        plt.show()

        return 1

    def esimate(self, bl_mean, std):

        ###first step, slope from the point to origin

        if self.esim_init.verbose:
            print(self.coefficient)

        mean_offsets = mean(self.offsets)

        #popt2, pcov2 = curve_fit(self.line_1,  [0, bl_mean-200], [0, std])
        slope = np.polyfit([0, bl_mean-mean_offsets], [0, std**2], 1)[0]

        if self.esim_init.show_graph:

            plt.figure()
            plt.scatter([0, bl_mean-mean_offsets], [0, std**2], marker='o')
            #plt.plot(bl_mean_array-offset, std_dev)
            plt.plot([0, bl_mean-mean_offsets], [x * slope for x in [0, bl_mean-mean_offsets]])

            plt.show()

            plt.figure()
            plt.plot(self.slopes, self.slopes_uncertainties)
            plt.show()


            ###estimate the uncertainty from 

        first_slope = self.slopes_uncertainties[0]
        last_slope = self.slopes_uncertainties[-1]

        slp_un = self.slopes_uncertainties
        slp_un.append(last_slope)

        slp = self.slopes
        slp.append(10000)

        f = interp1d(slp, slp_un)

        gain = slope / self.coefficient

        gain_unc = f(slope) / self.coefficient + (slope / (self.coefficient ** 2)) * self.coeff_uncertainty

        return gain, gain_unc


    def extract_gain(self):


        if self.esim_init.gain_extraction_method == "baseline":
            """
            baseline method
            ---------
            Extract the relationship between baseline shift and signal standard deviation to deduce a coefficient useful to extract gain
                
            """

            #Baseline.execute(self.esim_init, gain_min=2, gain_max=15) ###need to relocate
            pulse = Pulser(step=self.esim_init.t_step, pulse_type="none")
            evts = pulse.generate_all()


            gains = np.arange(self.gain_min, self.gain_max + 1, 1)

            fig, ax = plt.subplots()
            ax.set_title("Th vs Exp gain")
            ax.plot(gains, gains, label="theoretical")


            gain_2 = []
            noises = []
            exp_gain_unc = []

            line_tracker = 1

            last_val = []


            for i in np.linspace(1.0, 1.0, num=self.nb_lines):

                print(i)
                noises.append(i)

                # training
                coeff, coeff_uncertainty, offset_coeff, offsets = self.calculate_coeff(evts=evts, noi=i, line_nb_=line_tracker)

                # experience
                gains, slopes, slopes_uncertainties, offsets_ = self.loop_gain_bl(
                    evts=evts,
                    gain_min=self.gain_min,
                    gain_max=self.gain_max,
                    n1=self.n_exp_1,
                    n2=self.n_exp_2,
                    bk_min=self.bk_min,
                    bk_max=self.bk_max,
                    nsb_var=self.nsb_var,
                    noise_=i,
                    line_nb=line_tracker
                )

                # gains, slopes, slopes_uncertainties = self.loop_gain_bl(evts=evts, gain_min=2, gain_max=2,
                # n1=self.n_exp_1, n2=self.n_exp_2, bk_min=self.bk_min, bk_max=self.bk_max, nsb_var=self.nsb_var, noise_=i)

                exp_gain = []
                exp_gain_unc = []

                for i, q in enumerate(slopes):
                    unc_s = slopes_uncertainties[i]

                    exp_gain.append(q / coeff)
                    exp_gain_unc.append(unc_s / coeff + (q / (coeff * coeff)) * coeff_uncertainty)
                ax.plot(gains, exp_gain - offset_coeff, label="1")

                line_tracker += 1

                new_gain_exp = exp_gain - offset_coeff
                last_val.append(new_gain_exp[-1])

            ax.fill_between(gains, [a - b for a, b in zip(gains, exp_gain_unc)],[a + b for a, b in zip(gains, exp_gain_unc)],alpha=0.2)

            print("LAST UNCERT", exp_gain_unc[-1])
            ax.legend(loc="upper left")
            ax.set_xlabel("Gains")
            ax.set_ylabel("Gains")
            #now calculate the deviation

            plt.show()

            plt.figure()
            plt.scatter(last_val, np.zeros(len(last_val)))
            plt.hist(last_val, density=True, bins=10)

            ##calculate gaussian from histogram 

            data = norm.rvs(10.0, 2.5, size=500)
            
            mu, std = norm.fit(last_val)
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)

            plt.plot(x, p)

            x_values = [mu-std/2, mu+std/2]
            y_values = [0, 0]

            x_values2 = [mu-exp_gain_unc[-1]/2, mu+exp_gain_unc[-1]/2]
            y_values2 = [0, 0]
            plt.plot(x_values, y_values,'bo', linestyle="--")
            plt.plot(x_values2, y_values2,'ro', linestyle="--")

            plt.show()




        elif self.esim_init.gain_extraction_method == "pulse":
            """
            pulse method
            ---------
            Extract the relationship from pulse to pulse jitter (TODO)
                
            """
            Pulse.execute(self.esim_init)

        elif self.esim_init.gain_extraction_method == "debug":
            """
            debug method
            ---------
            Simple debug method to plot all the relationship between parameters
                
            """

            Debug.execute(self.esim_init)
        
        elif self.esim_init.gain_extraction_method == "under_c":
            """
            uncer c method
            ---------
            Plot the uncerestimation of the gain from theoretical gain on just one axis : Useful to readujust the slopes when there is an unwanted offset
                
            """

            Under_c.execute(self.esim_init)

        elif self.esim_init.gain_extraction_method == "bl_shift":
            
            """
            baseline shift method
            ---------
            Plot the relationship betwen the variance and baseline means : mostly useful for debugging
                
            """

            BL_shift.execute(self.esim_init)
                    
        elif self.esim_init.gain_extraction_method == "blstddev":

            """
            blstddev method
            ---------
            Useful to plot uncertainty vs standard deviation for crosschecks
                
            """

            BL_stddev.execute(self.esim_init)

        elif self.esim_init.gain_extraction_method == "deconvolution":
            ##write the deconvolution of the signal
            pulse = Pulser(step=self.esim_init.t_step, pulse_type="none")
            evts = pulse.generate_all()

            esim = TraceSimulation(
                ampSpec="../data/spe_R11920-RM_ap0.0002.dat",
                timeSpec="../data/bb3_1700v_timing.txt",
                pulseShape="../data/pulse_FlashCam_7dynode_v2a.dat",
            )

            evts_br, k_evts = esim.simulateBackground(evts)

            # pmt signal
            times, pmtSig, uncertainty_pmt = esim.simulatePMTSignal(evts_br, k_evts) #TODO : make uncertainty from the simulatePMTSignal, with ampdist.rvs(). For now sufficient


            eleSig, uncertainty_ele = esim.simulateElectronics(pmtSig, uncertainty_pmt, times)

            # adc signal
            stimes, samples, samples_unpro, uncertainty_sampled = esim.simulateADC(times, eleSig, uncertainty_ele, 1)

            #This part should be done all the time, even when it is loaded
            bl_mean, s_mean, std, std_unpro, bl_mean_uncertainty, bl_array, stddev_uncert_mean, stddev_mean, spike = self.esim_init.FPGA(stimes, samples, samples_unpro, uncertainty_sampled)

            square = np.repeat([0., 1., 0.], 10000)

            filter_ = [x+0.1 for x in esim.pulseShape[1]]

            plt.figure()
            plt.title("signal")
            plt.plot(stimes, samples)

            plt.figure()
            plt.title("filter")
            plt.plot(filter_)

            plt.figure()
            plt.title("convoluted")
            plt.plot(times, eleSig)
            plt.plot(times, pmtSig)

            

            ##denoising maybe ???


        return 1

    

    def calculate_coeff(self, evts, noi, line_nb_):
        """
        Calculate coeff method : derives the intrisic coefficient between true gain and slopes for this PMT
        The function acts as a training to derive the coeff which is then used to extract gain from any trace

        Parameters
        ----------
        evts - array_like
                PE time events coming from the pulser
        noi - float
                ADC noise
        line_nb - int
                index of the line being simulated. Useful to not reload the same trace twice

        Returns
        -------
        coeff
                intrisic coefficient
        coeff_ucnertainty
                uncertainty on said coefficient
        offset_coeff
                Potential offset when deriving the coefficient
        """

        #first extract the slopes of from the different gains
        gains, slopes, slopes_uncertainties, offsets = self.loop_gain_bl(evts=evts, gain_min=self.gain_min, gain_max=self.gain_max, 
                    n1=self.n_train_1, n2=self.n_train_2, bk_min=self.bk_min, bk_max=self.bk_max, nsb_var=self.nsb_var, noise_=noi, line_nb=line_nb_)


        self.slopes = slopes 
        self.slopes_uncertainties = slopes_uncertainties


        #Make sure we have train for the coeff (the bigger, the more precise)
        #Then calculate the coefficient

        if self.esim_init.show_graph:

            plt.figure()
            plt.plot(gains, slopes, 'bo')
            plt.fill_between(gains, [a - b for a, b in zip(slopes, slopes_uncertainties)], [a + b for a, b in zip(slopes, slopes_uncertainties)], alpha=0.2)
            plt.xlabel("Gain")
            plt.ylabel("Linear fit slopes")


        ####Here we have an error on coeff
        popt, pcov = curve_fit(self.line_1, gains, slopes, sigma=slopes_uncertainties)

        coeff = popt[0]
        offset_coeff = popt[1]
        coeff_uncertainty = pcov[0,0]**0.5

        if self.esim_init.show_graph:

            plt.plot(gains, [x * coeff + offset_coeff for x in gains])


        return coeff, coeff_uncertainty, offset_coeff, offsets

    

    def loop_gain_bl(self, evts, gain_min, gain_max, n1, n2, bk_min, bk_max, nsb_var, noise_=1.0, line_nb=1):
        """
        Loops over both gains and background rate to extact the slopes

        Parameters
        ----------
        evts - array_like
                Pe times coming from the pulser
        gain_min - float
                Minimum gain from which to perform the loop
        gain_max - float
                Maxmimum gain from which to perform the loop
        n1 - int
                number of loop for gain space
        n2 - int   
                number of loop for background rate space
        bk_min - float
                minimum background rate from which to perform the loop
        bk_max - float
                maximum background rate from which to perform the loop
        nsb_var - float
                Variation of the background rate in percent per second
        noise_ - float
                stddev of the ADC noise
        line_nb - int
                index of the current line

        Returns
        -------
        gains - array_like
                True gains
        slopes - array_like
                Slopes of stddev vs bl_shift for gains
        slopes_uncertainties - array_like
                uncertainties of said slopes
        """


        slopes = []
        slopes_uncertainties = []
        gains = []
        offsets = []

        if self.esim_init.show_graph:
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
            for i in np.logspace(bk_min, bk_max, num=n2):
            #for i in np.logspace(bk_min, bk_max, num=n2):

                #we need to add random evts that follow a negative exponential for the background rate

                ####Generating the traces beforehand###
                #look if a trace with gain=j, background_rate=i, max_nsb_var= ... and noise= ... exists
                #if so loads it, if not generate it.
                #Each generation is saved in a file
                #When the generation method is changed, just remove all files from /exports

                if self.esim_init.verbose:
                    print("background",i)
                    print('exports/B='+str(i)+';G='+str(j)+';V='+str(nsb_var)+';N='+str(noise_)+'line=' + str(line_nb) + '.txt')

                if self.load_files and os.path.exists('exports/B='+str(i)+';G='+str(j)+';V='+str(nsb_var)+';N='+str(noise_)+'line=' + str(line_nb) + '.npy'):
                    #load it
                    if self.esim_init.verbose:
                        print("Loading file..")

                    with open('exports/B='+str(i)+';G='+str(j)+';V='+str(nsb_var)+';N='+str(noise_)+'line=' + str(line_nb) + '.npy', 'rb') as f:

                        stimes = np.load(f)
                        samples = np.load(f)
                        samples_unpro = np.zeros(samples.shape)
                        uncertainty_sampled = np.load(f)
                        bl_mean = np.load(f)
                        std = np.load(f)
                        stddev_mean = np.load(f)
                        spike = np.load(f)
                        s_mean = np.load(f)
                        std_unpro = np.load(f)
                        bl_mean_uncertainty = np.load(f)
                        bl_array = np,load(f)
                        stddev_uncert_mean = np.load(f)
                        skew = np.load(f)
                        adc_noise = np.load(f)
                        true_gain = np.load(f)
                        true_background_rate = np.load(f)
                        ####load the parameters the classical way such as not to reload everything when I change one parameter
                    

                else:
                    #generate it
                    esim = TraceSimulation(
                        ampSpec="../data/spe_R11920-RM_ap0.0002.dat",
                        timeSpec="../data/bb3_1700v_timing.txt",
                        pulseShape="../data/pulse_FlashCam_7dynode_v2a.dat",
                        background_rate = i,
                        gain=j,
                        max_nsb_var=nsb_var,
                        noise=noise_,
                    )

                    evts_br, k_evts = esim.simulateBackground(evts)

                    # pmt signal
                    times, pmtSig, uncertainty_pmt = esim.simulatePMTSignal(evts_br, k_evts) #TODO : make uncertainty from the simulatePMTSignal, with ampdist.rvs(). For now sufficient


                    eleSig, uncertainty_ele = esim.simulateElectronics(pmtSig, uncertainty_pmt, times)

                    # adc signal
                    stimes, samples, samples_unpro, uncertainty_sampled = esim.simulateADC(times, eleSig, uncertainty_ele, line_nb)

                    #This part should be done all the time, even when it is loaded
                    bl_mean, s_mean, std, std_unpro, bl_mean_uncertainty, bl_array, stddev_uncert_mean, stddev_mean, spike, skew = esim.FPGA(stimes, samples, samples_unpro, uncertainty_sampled, line_nb, True)

                bl_mean_array.append(bl_mean)
                bl_mean_uncer_array.append(bl_mean_uncertainty)

                s_mean_array.append(s_mean)
                freq.append(i)
                std_dev.append(stddev_mean**2)
                std_dev_unpro.append(std_unpro)
                theoretical.append(self.esim_init.singePE_area*self.esim_init.gain*self.esim_init.background_rate* 1e-9 + self.esim_init.offset)
                ratio_bl_exp.append(bl_mean/(self.esim_init.singePE_area*self.esim_init.gain*self.esim_init.background_rate* 1e-9 + self.esim_init.offset))
                
                if self.esim_init.show_signal_graphs:
                    plt.figure()
                    plt.title("Simulated ADC output")
                    plt.plot(stimes + self.esim_init.plotOffset, samples, label="Background rate : 1e9 [Hz]")
                    plt.plot(stimes + self.esim_init.plotOffset, np.ones(len(stimes))*bl_mean, label="Baseline")
                    #plt.fill_between(stimes + self.esim_init.plotOffset, [a - b for a, b in zip(samples, uncertainty_sampled)], [a + b for a, b in zip(samples, uncertainty_sampled)], alpha=0.2)
                    plt.xlabel("Time/ns")
                    plt.ylabel("ADC output/LSB")
                    plt.legend(loc="upper left")
                    plt.show()


            #Lets calculate the uncertainty on offset :
            #Uncertainty on the fit

            popt, pcov = curve_fit(self.line_1, freq, bl_mean_array, sigma=bl_mean_uncer_array)
            offset = popt[1]
            offset_uncertainty = pcov[1,1]**0.5

            #Try to reimplement the classical curve fit with the self variable
            if self.esim_init.slope_method == "odr":

                #Substraction of uncertainty
                quad_model = odr.Model(self.line_)

                data = odr.RealData(bl_mean_array - offset, std_dev, sx=bl_mean_uncer_array + offset_uncertainty, sy=bl_mean_uncer_array + offset_uncertainty)
                odr_ = odr.ODR(data, quad_model, beta0=[1., 0.])

                out = odr_.run()

                popt2 = out.beta
                perr2 = out.sd_beta
                slope = popt2[0]
                slope_error = perr2[0]

            elif self.esim_init.slope_method == "classical":
                popt2, pcov2 = curve_fit(self.line_1,  bl_mean_array - offset, std_dev, sigma=bl_mean_uncer_array + offset_uncertainty)

                slope = popt2[0]
                offset_ = popt2[1]

                slope_error = pcov2[0,0]**0.5

            if self.esim_init.show_graph == True:

                if self.esim_init.verbose:
                    print("bl_mean", bl_mean_uncer_array)

                ###Do the plt annotate

                normalizes_blarray = bl_mean_array-offset

                annotation = [str(x) for x in freq]

                ##trigger a 20/30khz -> 1e4 hz

                for i, label in enumerate(annotation):
                    plt.annotate(label, (normalizes_blarray[i], std_dev[i]))


                ratio = []
                for i in range(len(normalizes_blarray)):
                    ratio.append(normalizes_blarray[i]/std_dev[i])


                #plt.scatter(np.zeros(len(ratio)), ratio)
                
                plt.errorbar(normalizes_blarray, std_dev, xerr=bl_mean_uncer_array + offset_uncertainty, yerr=bl_mean_uncer_array + offset_uncertainty, fmt='o')
                #plt.plot(np.log10(normalizes_blarray), np.log(std_dev))



                #plt.plot(bl_mean_array-offset, std_dev)
                
                #plt.plot(normalizes_blarray, [x * slope for x in normalizes_blarray], label="gain="+str(j))
                
            ###########debuging right here
            #plt.plot(freq, std_dev)

            if self.esim_init.verbose:
                print("freq",)

            slopes.append(slope)
            slopes_uncertainties.append(slope_error)

            offsets .append(offset)

        if self.esim_init.show_graph:
            plt.xlabel("Baseline Shift [LSB]")
            plt.ylabel("Variance [LSB]")
            #plt.title("Variation of signal Variance vs Baseline shift for different background rates")
            plt.legend(loc="upper left")

        return gains, slopes, slopes_uncertainties, offsets
