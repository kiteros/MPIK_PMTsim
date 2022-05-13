#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm, rv_histogram, randint, poisson, expon, exponnorm, skew
from scipy.signal import resample
from pulser import Pulser
import statistics
import scipy.integrate as integrate
import math 
import sys
from scipy.optimize import curve_fit
import scipy.special as sse



class TraceSimulation:
    """
    Class to simulate PMT and ADC response to photo electron events.

    Use `simulateAll()` to simulate full electronics chain or methods for
    individual simulations if intermiediate results are required.

    Attributes
    ----------
    f_sample : float
            sampling rate of ADC in 1/ns
    t_sample : float
            sampling time of ADC in ns
    oversamp : int
            oversampling factor
    t_step : float
            time step for simulation in ns (`t_step=t_sample/oversamp`)
    t_pad : float
            padding before first and after last event in ns
    offset : int
            ADC offset in LSB
    gain : int
            ADC gain (scale factor of normalized amplitude) in LSB
    noise : float
            standard deviation of gaussian noise in LSB
    jitter : int or None
            offset of ADC samples relative to simulation times, `None` for random jitter
    timeSpec : tuple of array_like
            PE delay times in ns and their probabilities
    timeDist : rv_histogram
            distribution of PE delay times
    ampSpec : tuple of array_like
            PE amplitudes in normalized units and their probabilities
    ampsDist : rv_histogram
            distribution of PE amplitudes
    pulseShape : tuple of array_like
            times in ns and amplitudes in normalized units of pulse shape
    plotOffset : float
            time offset for nice plotting in ns
    debugPlots : bool
            plot intermediate results if `True`
    background_rate_method : string
            Defines the distribution for the BR ("poisson" for a variate per sample, "exponential" for a negative exponential distribution of time
            delays for black counts). BC can come from thermionic or night sky
    transit_time_jitter : float
            gaussian variation of arrival times on the anode of PE
    background_rate : float
            Rate of evts coming from the background
    br_mu : float
            Mu parameter of the poisson distribution of the background rate
    br_lamda_exp : float
            Lamda parameter of the exponential distribution
    show_graph : bool
            Show final graphs of the simulation
    no_signal_duration : float
            Duration in ns of the signal if no evts is generated by generateBackground()
    remove_padding : bool
            Removing the padding of t_pad to allow for a more accurate baseline following
    max_nsb_var : float
            Maximum variation of the background freqency per second in percent (percent/s)
    nsb_fchange : float
            Sampling of the NSB variation in Hz
    gain_extraction method : string
            "baseline" : plot stddev vs baseline and extract coefficient from slopes for multiple background_rate
            "pulse" : From the pulsed light
    slope_method : string
            classical : np.fit, uncertainty on only one side
            odr : Orthogonal descent regretion with x and y uncertainties

    """

    def __init__(
        self,
        f_sample=0.25,
        oversamp=40,
        t_pad=200,
        offset=200,
        noise=0.8, #ADC noise : 0.5 - 1.5 LSB

        #Gain from 2 to 15 LSB
        gain=15,# in fadc_amplitude in CTA.cfg; ALSO : gain=7.5e5 from CTA-ULTRA5-small-4m-dc.cfg
        jitter=None,
        debugPlots=False,
        timeSpec=None,
        ampSpec=None,
        pulseShape=None,

        ##Flashcam parameters
        transit_time_jitter = 0.75, #From CTA.cfg, also : 0.64 Obtained from pmt_specs_R11920.cfg
        background_rate_method = "poisson", #poisson, exponential
        #NSB from 0..2GHz
        background_rate = 1e7, #Hz
        show_graph = False,
        no_signal_duration = 1e4, #in ns
        remove_padding = True,
        max_nsb_var = 0.1, #Maximum variation of the NSB per second
        nsb_fchange = 1e6, #Hz frequency for the implemented variation of nsb var rate
        gain_extraction_method = "baseline", #pulse, baseline, debug, under_c, blstddev, bl_shift

        slope_method = "odr", #classical, odr
        show_signal_graphs = False,
        verbose = False,

        ps_mu = 15.11,
        ps_amp = 22.0,
        ps_lambda = 0.0659,
        ps_sigma = 2.7118,

        pulse_size = 150,
        pulse_sampling = 1500,
        
    ):
        """
        Initializes instances of TraceSimulation class.

        `timeSpec`, `ampSpec` and `pulseShape` can have special values:
                'None' -- simulates gaussian spectra/pulse shape
                string -- loads spectra/pulse shape from file with given name
        """
        self.f_sample = f_sample
        self.t_sample = 1 / f_sample
        self.oversamp = oversamp
        self.t_step = self.t_sample / oversamp
        self.t_pad = t_pad
        self.offset = offset
        self.noise = noise
        self.gain = gain
        self.jitter = jitter
        self.debugPlots = debugPlots
        self.transit_time_jitter = transit_time_jitter
        self.background_rate_method = background_rate_method
        self.background_rate = background_rate
        self.show_graph = show_graph
        self.no_signal_duration = no_signal_duration
        self.remove_padding = remove_padding
        self.max_nsb_var = max_nsb_var
        self.nsb_fchange = nsb_fchange
        self.gain_extraction_method = gain_extraction_method
        self.slope_method = slope_method
        self.show_signal_graphs = show_signal_graphs
        self.verbose = verbose
        self.ps_mu = ps_mu
        self.ps_amp = ps_amp
        self.ps_lambda = ps_lambda
        self.ps_sigma = ps_sigma
        self.pulse_size = pulse_size
        self.pulse_sampling = pulse_sampling

        np.set_printoptions(threshold=sys.maxsize)

        #calculate the poissonian lamda and exp mu
        #We don't take into account the variation of the lamdas and mu through time for the uncertainties

        self.lamda = self.t_step * self.background_rate * 1e-9#From the def E(poisson) = lamda

        self.mu = self.background_rate * 1e-9#From the def E(exp) = 1/mu

        #Loading the time spectrum
        if isinstance(timeSpec, str):
            self.timeSpec = np.loadtxt(timeSpec, unpack=True)
            if self.timeSpec.ndim == 1:
                binCnt = int((self.timeSpec.max() - self.timeSpec.min()) / self.t_step)
                hist, bins = np.histogram(self.timeSpec, binCnt, density=True)
                self.timeSpec = (bins[:-1], hist)
                self.timeDist = rv_histogram((hist, bins))
        elif isinstance(timeSpec, scipy.stats._distn_infrastructure.rv_frozen):
            self.timeSpec = timeSpec
        elif isinstance(timeSpec, float):
            #Jitter is very small (<ns) so we can approximate by a gaussian : We don't load the time spectrum anymore
            self.timeSpec = self.simulateTimeSpectrum(t_sig = self.transit_time_jitter) 
        else :
            self.timeSpec = self.simulateTimeSpectrum()


        #load ampspec from file
        if ampSpec == None:
            self.ampSpec = self.simulateAmplitudeSpectrum()
            # TODO histogram?
        elif type(ampSpec) is str:
            #self.ampSpec = [np.loadtxt(ampSpec, unpack=True)[0],np.loadtxt(ampSpec, unpack=True)[1]]
            self.ampSpec = np.loadtxt(ampSpec)[:,:2].T

        else:
            self.ampSpec = ampSpec

        # load pulse shape from file or simulate
        if pulseShape == None:
            self.pulseShape = self.simulatePulseShape(ps_amp, ps_lambda, ps_sigma, ps_mu)
            print("simulating pulse shape")
        elif type(pulseShape) is str:
            ps = np.loadtxt(pulseShape, unpack=True)
            # ensure correct sampling
            step = ps[0][1] - ps[0][0]
            ps, t = resample(ps[1], int(step / self.t_step * ps[1].shape[0]), ps[0])
            self.pulseShape = (t, ps)
            # offset from center (for plots only)
            self.plotOffset = (t[-1] + t[0]) / 2

            ##calculate the parameters of the pulse shape
            ###fit it
            print(self.fit_pulseshape(t, ps))
        else:
            self.pulseShape = pulseShape
        self.singePE_area = self.integrateSignal(self.pulseShape[0], self.pulseShape[1])
        

        # get distributions of spectra
        sx = self.timeSpec[0]
        bins = np.append(sx, sx[-1] + sx[1] - sx[0])
        self.timeDist = rv_histogram((self.timeSpec[1], bins))
        sx = self.ampSpec[0]
        bins = np.append(sx, sx[-1] + sx[1] - sx[0])
        self.ampDist = rv_histogram((self.ampSpec[1], bins))

        self.pulseMean, self.pulseStddev = self.statsCalc(self.pulseShape[0], self.pulseShape[1], self.ampDist)
        self.ampMean, self.ampStddev = self.statsCalc(self.ampSpec[0], self.ampSpec[1], self.ampDist)
        self.ampDist_drift = self.ampDist.mean()#1.025457559561722


        # figures for debugging
        if self.debugPlots:
            plt.figure()
            plt.plot(*self.pulseShape)
            plt.title("Pulse shape")
            plt.xlabel("t/ns")
            plt.ylabel("A/au")
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.plot(*self.timeSpec)
            plt.title("Time spectrum")
            plt.xlabel("t/ns")
            plt.ylabel("Probability")
            plt.subplot(1, 2, 2)
            plt.plot(*self.ampSpec)
            plt.title("Amplitude spectrum")
            plt.xlabel("A/au")
            plt.ylabel("Probability")

    def integrateSignal(self, times, signal):
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

        sum_ = 0
        for i in signal:
            sum_ += i*self.t_step # maybe wrong
        return sum_

    def calculateBL_underest(self, background_rate):

        #Obsolete
        return math.log10(background_rate) * 0.4848922809933432 -1.4096232853103452


    def statsCalc(self, times, signals, dist):

        """
        Calculate various statistical elements

        Parameters
        ----------

        Returns
        -------
        """

        mean = dist.mean()
        #stddev = np.std(signals-np.ones(signals.shape)*mean)
        stddev = math.sqrt(dist.var())
        return mean, stddev

    def simulatePETimes(self, npe=10, t_mu=0, t_sig=300):
        """
        Simulates gaussian photo electron arrival times.

        Parameters
        ----------
        npe - int
                number of photo electrons
        t_mu - float
                mean time in ns
        t_sig - float
                standard deviation in ns

        Returns
        -------
        ndarray
                list of times in ns
        """
        return norm.rvs(t_mu, t_sig, npe)

    def simulateTimeSpectrum(self, t_mu=0, t_sig=1):
        """
        Simulates a gaussian time spectrum.

        Parameters
        ----------
        t_mu - float
                mean time in ns
        t_sig - float
                standard deviation in ns

        Returns
        -------
        tuple of ndarray
                times in ns and their probabilities
        """
        #Modification to center the gaussian around 0
        #tsx_onecentered = np.arange( int((t_mu + 3 * t_sig) / self.t_step)) * self.t_step
        tsx = np.arange( int(((t_mu + 3 * t_sig) / self.t_step)) / 2.0) * self.t_step
        tsx_negative = np.arange( int(((t_mu + 3 * t_sig) / self.t_step)) / 2.0) * self.t_step * (-1)
        tsx_negative = tsx_negative[::-1]
        tsx_negative = tsx_negative[:-1]
        tsx_zerocentered = np.concatenate((tsx_negative, tsx), axis=None)

        return tsx_zerocentered, norm.pdf(tsx_zerocentered, t_mu, t_sig)

    def simulateAmplitudeSpectrum(self, a_mu=0, a_sig=1):
        """
        Simulates a gaussian amplitude spectrum.

        Parameters
        ----------
        a_mu - float
                mean amplitude (set to 1 for nomalized output)
        a_sig - float
                standard deviation

        Returns
        -------
        tuple of ndarray
                amplitudes and their probabilities
        """
        asx = np.arange(int((a_mu + 3 * a_sig) / 0.01)) * 0.01
        return asx, norm.pdf(asx, a_mu, a_sig)

    def expnorm_fit(self, x, A, l, s, m):
        return A*0.5*l*np.exp(0.5*l*(2*m+l*s*s-2*x))*sse.erfc((m+l*s*s-x)/(np.sqrt(2)*s)) # exponential gaussian


    def fit_pulseshape(self, x, y):
        
        vals = np.linspace(0, 100, num=len(x))


        max_value = x[-1]
        popt, pcov = curve_fit(self.expnorm_fit, vals, y)
        #print(ps[0])
        new_mu = (popt[3]/100)*max_value
        new_lamda = popt[1]/(max_value/100)
        new_amplitude = popt[0]*(max_value/100)
        new_sigma = popt[2]*(max_value/100)

        ####calculate bandwidth
        risetime = new_sigma/(math.erf(0.8)*np.sqrt(2.0/np.pi))
        bandwidth = new_lamda/(2.0 * np.pi)

        return new_amplitude, new_lamda, new_sigma, new_mu


    def simulatePulseShape(self, A=22.0, l=0.0659, s=2.7118, m=15.116):
        """
        Simulates a gaussian pulse shape.

        Parameters
        ----------
        t_mu - float
                peak time in ns
        t_sig - float
                pulse width in ns

        Returns
        -------
        tuple of ndarray
                times in ns and amplitudes in normalized units
        """
        """Gaussian pulse with filtered bandwidth in GHz and 10%-90% risetime of the *unfiltered* pulse in nanoseconds."""
        
        K = 1/(l*s)
        x = np.linspace(0, self.pulse_size, num=self.pulse_sampling)
        return x, A*exponnorm.pdf(x, K, loc=m, scale=s)
        
    def simulatePMTSignal(self, peTimes, k_evts):
        """
        Simulates PMT signal based on photo electron times.

        Parameters
        ----------
        peTimes - array_like
                list of photo electron arrival times in ns

        k_evts - array_like
                number of PE per peTime, the number are repeated in order to match the peTime array

        Returns
        -------
        tuple of ndarray
                times in ns and simulated signal
        """
        tot_pe = sum(k_evts)

        # make discrete times
        if len(peTimes) > 0:
            t_min = peTimes.min() - self.t_pad
            t_max = peTimes.max() + self.t_pad
        else :
            t_min = (-1) * self.t_pad
            t_max = self.no_signal_duration + self.t_pad

        times = np.arange((int(t_max - t_min) / self.t_step)) * self.t_step + t_min
        # make signal
        signal = np.zeros(times.shape)

        #self.uncertainty_averaged = self.ampStddev * self.lamda + self.ampMean*math.sqrt(self.lamda)
        self.uncertainty_averaged = self.lamda * self.ampMean

        uncertainty = np.ones(times.shape)*self.uncertainty_averaged

        #We need a way to know the number of Pe with same time
        #This information is in k_evts, need to be careful with indexes

        self.n_steps = (t_max - t_min ) / self.t_step

        if self.background_rate_method == "poisson":

            for i, t in enumerate(peTimes):

                amp_rvs = self.ampDist.rvs() / self.ampDist_drift

                t += self.timeDist.rvs()
                signal[int((t - t_min) / self.t_step)] += amp_rvs

        elif self.background_rate_method == "exponential":
            print("todo")
                
        return times, signal, uncertainty

    def simulateElectronics(self, signal, uncertainty, times):
        """
        Simulates effect of electronics on PMT signal.

        Parameters
        ----------
        signal - array_like
                signal from PMT (use result of `simulatePMTSignal()`)

        Returns
        -------
        ndarray
                simulated signal
        """
        if self.show_signal_graphs:

            plt.figure()
            plt.plot(times, signal, label="signal")
            plt.plot(times, uncertainty, label="uncertainty")
            #plt.scatter(np.arange(0,self.no_signal_duration, self.t_step), np.zeros(np.arange(0,self.no_signal_duration, self.t_step).shape),  label="step")
            plt.legend(loc="upper left")
            plt.show()

        #We can convolve with a constant, see overleaf for equation
        #uncertainty = np.convolve(signal, self.pulseStddev, "same") + np.convolve(uncertainty, self.pulseShape[1], "same")
        #uncertainty = np.repeat(uncertainty[0]*self.singePE_area, len(uncertainty))  #np.convolve(uncertainty, self.pulseShape[1], "same")
        #uncertainty = uncertainty[0] * self.singePE_area + self.ampMean * self.pulseStddev * self.lamda * self.no_signal_duration * 1e-9

        uncertainty = np.sqrt(np.convolve(uncertainty, self.pulseShape[1], "same"))

        return np.convolve(signal, self.pulseShape[1], "same"), uncertainty#np.repeat(uncertainty, len(signal))

    def simulateADC(self, times, signal, uncertainty, line_nb):
        """
        Simulates ADC out based on electronics signal.

        Parameters
        ----------
        times - array_like
                times of signal in ns
        signal - array_like
                amplitudes of signal in normalized units (use result of `simulateElectronics`)

        Returns
        -------
        tuple of ndarray
                times in ns and simulated output in LSB
        """
        jitter = self.jitter
        if jitter == None:
            jitter = randint.rvs(0, self.oversamp)  # TODO random size?
        stimes = times[jitter :: self.oversamp]

        
        samples = signal[jitter :: self.oversamp] * self.gain + norm.rvs(self.offset, self.noise, stimes.shape)
        uncertainty_sampled = uncertainty[jitter :: self.oversamp] * self.gain + self.noise #Only the stddev of noise impacts

        samples = samples.astype(int) #numpy round/add 0.5 before

        samples_unpro = signal[jitter :: self.oversamp]
        samples_unpro = samples_unpro.astype(int)

        #Removing padding for easier readability
        if self.remove_padding:
            extra_padding = 10
            stimes = stimes[int(self.t_pad // 4)+extra_padding:]
            stimes = stimes[:len(stimes) - int(self.t_pad // 4) - extra_padding]
            samples = samples[int(self.t_pad // 4)+extra_padding:]
            samples = samples[:len(samples) - int(self.t_pad // 4) - extra_padding]

            samples_unpro = samples_unpro[int(self.t_pad // 4)+extra_padding:]
            samples_unpro = samples_unpro[:len(samples_unpro) - int(self.t_pad // 4) - extra_padding]

            uncertainty_sampled = uncertainty_sampled[int(self.t_pad // 4)+extra_padding:]
            uncertainty_sampled = uncertainty_sampled[:len(uncertainty_sampled) - int(self.t_pad // 4) - extra_padding]

        if self.show_signal_graphs:
            plt.figure()
            plt.plot(stimes, samples, label="signal")
            plt.plot(stimes, uncertainty_sampled+np.mean(samples), label="uncertainty on signal")
            plt.plot(stimes, np.repeat(np.mean(samples), uncertainty_sampled.shape), label="mean signal")
            plt.plot(stimes, np.repeat(np.std(samples), uncertainty_sampled.shape)+np.mean(samples), label="stddev signal")
            #plt.scatter(np.arange(0,self.no_signal_duration, self.oversamp), np.zeros(np.arange(0,self.no_signal_duration, self.oversamp).shape),  label="step")
            plt.legend(loc="upper left")
            plt.show()



        
            

        ##Here uncertainty sampled is just a repetition of the same uncertainty
        return stimes, samples, samples_unpro, uncertainty_sampled

    def simulateBackground(self, evts):
        """
        Simulate random emmission of PE either from the inside of the PMT (thermionic, rogue electron), or from photons arriving
        from background sources (eg. night sky)

        Parameters
        ----------
        evts - array_like times of PE in ns

        Returns
        -------
        array_like of new times of PE in ns
        
        """

        #Implement change of nsb per second
        var_time = (1/self.nsb_fchange) * 1e9 #ns

        #first value of the background rate for reference
        bg_ref = self.background_rate

        #We define an array storing the uncertainty at every stime

        if self.background_rate_method == "exponential":
            if len(evts) > 0:
                t_min = evts.min() - self.t_pad
                t_max = evts.max() + self.t_pad
                evts_list = evts.tolist()
            else :
                t_min = (-1) * self.t_pad
                t_max = self.no_signal_duration + self.t_pad
                evts_list = []
            sxap = t_min
            time_delay = 0
            cum_time = 0

            #convert np.array() to python list
            
            while sxap < t_max:
                sxap += expon.rvs(scale = 1/abs(mu)) #scale = 1/mu
                #print(expon.rvs(scale = 1/mu))
                time_delay += sxap
                cum_time += sxap
                evts_list.append(sxap)

                if time_delay > var_time:
                    self.background_rate += norm.rvs(0, ((self.max_nsb_var * var_time * bg_ref)/(1e9))/(2*math.sqrt(2*math.log(2))), 1)[0]
                    mu = self.background_rate * 1e-9#From the def E(exp) = 1/mu
                    time_delay = 0

            evts = np.array(evts_list)


        elif self.background_rate_method == "poisson":

            new_background_rate = self.background_rate

            if len(evts) > 0:
                t_min = evts.min() - self.t_pad
                t_max = evts.max() + self.t_pad
                #convert np.array() to python list
                evts_list = evts.tolist()
            else :
                t_min = (-1) * self.t_pad
                t_max = self.no_signal_duration + self.t_pad
                evts_list = []

            
            n_steps_var_time = int(var_time // self.t_step)
            n_step_tot = int((t_max - t_min) // self.t_step)
            n_intervals = int(n_step_tot // n_steps_var_time)

            #Make a security if n_step_var_time is bigger than the total time TODO


            k_evts = []

            #We divide the whole signal in intervals with nsb_fchange and change the br for each interval
            for j in range(n_intervals):
                #recalculate the lamda each time

                for i, q in enumerate(poisson.rvs(abs(self.lamda), size=n_steps_var_time)):
                    # * operator create q times the same list
                    evts_list.extend([t_min + j*var_time + i * self.t_step] * q)
                    k_evts.extend([q] * q)
                    #So the time linked to i has q values
                    #We append to the uncertainty, index i (implicit)

                new_background_rate += norm.rvs(0, ((self.max_nsb_var * var_time * bg_ref)/(1e9))/(2*math.sqrt(2*math.log(2))), 1)[0]
                self.lamda = self.t_step * new_background_rate * 1e-9#From the def E(poisson) = lamda

            #fill the last interval
            for i, q in enumerate(poisson.rvs(abs(self.lamda), size=int(abs((t_max - n_intervals*var_time) // self.t_step)))):
                evts_list.extend([t_min + var_time*n_intervals + i * self.t_step] * q)
                k_evts.extend([q] * q)

            evts = np.array(evts_list)
        
        return evts, k_evts


    def stddev(self, arr, ref, mean_signal):

        #Homemade standard deviation

        deviations = [(mean_signal + x - ref) ** 2 for x in arr]
        deviations = sum(deviations) / len(arr)

        return math.sqrt(deviations)

    def FPGA(self, times, signal, samples_unpro, uncert, line_nb, save_graph):

        bl = int(self.singePE_area*self.gain*self.background_rate*1e-9+self.offset)
        print(bl)
        bl_array = []
        uncert_bl_mean_R = 0 #region method

        for i in range(len(times)):
            if signal[i] > bl:
                bl += 0.125

            elif signal[i] < bl:
                bl -= 0.125

            if bl > signal[i] - uncert[i] and bl < signal[i] + uncert[i]:
                #bl is inside the uncertainty region
                uncert_bl_mean_R += 0.125

            bl_array.append(bl)

        bl_unpro = samples_unpro[0]
        bl_array_unpro = []

        for i in range(len(times)):
            if samples_unpro[i] > bl_unpro:
                bl_unpro += 0.125

            elif samples_unpro[i] < bl_unpro:
                bl_unpro -= 0.125

            bl_array_unpro.append(bl_unpro)


        uncert_bl_mean = statistics.fmean(uncert)
        if self.verbose:
            print("before uncert transform", uncert_bl_mean)

        ##recalculate the right uncert on bl from the smoothing coefficient

        self.smoothing_coeff_offset = np.float32(-0.14140167346889987) #maybe should depend on the the gain : TODO

        transformed_signal = math.log10(uncert_bl_mean) + (self.smoothing_coeff_offset)
        transformed_signal = (10 ** transformed_signal) 

        transformed_signal = transformed_signal / math.sqrt(len(uncert))

        if self.verbose:
            print("after uncert transform", transformed_signal)

        bl_mean = statistics.fmean(bl_array)
        s_mean = statistics.fmean(signal)

        

        #stddev_uncert_baseline = np.std(bl_array-uncert)
        #stddev_uncert_baseline_mean = np.stddev(arr=uncert, ref=bl_mean, mean_signal=s_mean)
        stddev_uncert_mean = statistics.fmean(uncert) #self.stddev(arr=uncert, ref=s_mean, mean_signal=s_mean) ##rightest one

        stddev_baseline = np.std(bl_array) #to export
        #stddev_baseline_mean = np.std(np.ones(len(signal))*bl_mean-signal)
        stddev_mean = np.std(signal, ddof=1) #rightest one

        #We want to know the relationship between stddev_baseline and stddev_mean

        """OUTPUTS
            1) baseline mean, +self.bl_underestimation can be added¨, it is bl_mean
            2) Signal mean
            3) standard deviation baseline
            4) This is wrong and needs to be changed
            5) Transformed signal is the uncertainty output of the bl for a given signal uncertainty (bl_mean_uncertainty)

        """

        ####to remove after, this is a safe way to know uncertainty
        transformed_signal = np.std(bl_array)



        ##############Calculate the spike coefficient

        spike = 0

        for j in range(len(signal)-1):
            spike += (signal[j]-signal[j+1])**2


        ####print the signal : 
        #Save it np.save()

        stddev_unpro = np.std(bl_array_unpro-samples_unpro)




        ####Calculate the skew\

        skew_ = skew(signal)
        if self.verbose:
            print("skew", skew_)

        if save_graph:

            with open('exports/B='+str(self.background_rate)+';G='+str(self.gain)+';V='+str(self.max_nsb_var)
                +';N='+str(self.noise)+'line=' + str(line_nb) + '.npy', 'wb') as f:
                
                np.save(f, times)
                np.save(f, signal)
                np.save(f, uncert)
                np.save(f, bl_mean)
                np.save(f, stddev_baseline)
                np.save(f, stddev_mean)
                np.save(f, spike)
                np.save(f, s_mean)
                np.save(f, stddev_unpro)
                np.save(f, transformed_signal)
                np.save(f, bl_array)
                np.save(f, stddev_uncert_mean)
                np.save(f, skew_)
                np.save(f, self.noise)
                np.save(f, self.gain)
                np.save(f, self.background_rate)




        return bl_mean, s_mean, stddev_baseline, stddev_unpro, transformed_signal, bl_array, stddev_uncert_mean, stddev_mean, spike, skew_


    def deconvolve(self):
        return 1

    def simulateAll(self, peTimes):
        """
        Simulates ADC output based on photo electron times.

        Parameters
        ----------
        peTimes - array_like
                list of photo electron arrival times in ns

        Returns
        -------
        tuple of ndarray
                times in ns and simulated ADC output in LSB
        """
        # simulate pmt
        times, signal = self.simulatePMTSignal(peTimes)
        # convolve with pulse shape
        signal = self.simulateElectronics(signal)
        # make samples
        stimes, samples = self.simulateADC(times, signal)
        # debug plots
        if self.debugPlots:
            plt.figure()
            plt.scatter(peTimes, np.zeros(peTimes.shape))
            plt.plot(times + self.plotOffset, signal)
            plt.xlabel("t/ns")
            plt.ylabel("A/au")
            plt.title("PMT signal")
            plt.figure()
            plt.plot(stimes, samples)
            plt.title("ADC output")
            plt.xlabel("t/ns")
            plt.ylabel("A/LSB")

        return stimes, samples


def example_usage():
    from matplotlib import pyplot as plt
    # plot spectra

    

    
if __name__ == "__main__":
    example_usage()
