#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm, rv_histogram, randint, poisson, expon
from scipy.signal import resample
from pulser import Pulser
import statistics
import scipy.integrate as integrate
import math 



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
    electronic_noise_amplitude : float
            Amplitude of the noise produced by electronics
    br_amplitude : float
            Amplitude of the bacgkground rate (not implemented yet)
    br_mu : float
            Mu parameter of the poisson distribution of the background rate
    br_lamda_exp : float
            Lamda parameter of the exponential distribution
    """

    def __init__(
        self,
        f_sample=0.25,
        oversamp=40,
        t_pad=200,
        offset=200,
        noise=0.8, #ADC noise : 0.5 - 1.5 LSB

        #Gain from 2 to 15 LSB
        gain=10,# in fadc_amplitude in CTA.cfg; ALSO : gain=7.5e5 from CTA-ULTRA5-small-4m-dc.cfg
        jitter=None,
        debugPlots=False,
        timeSpec=None,
        ampSpec=None,
        pulseShape=None,

        ##Flashcam parameters
        transit_time_jitter = 0.75, #From CTA.cfg, also : 0.64 Obtained from pmt_specs_R11920.cfg
        #electronic_noise_amplitude = 4.0, #From CTA.cfg, high gain

        background_rate_method = "poisson", #poisson, exponential
        #TODO : Issue with the exponential

        #NSB from 0..2GHz
        background_rate = 1e7, #Hz
        show_graph = True,

        no_signal_duration = 1e5, #in ns
        remove_padding = True,

        max_nsb_var = 1e1, #Maximum variation of the NSB per second
        nsb_fchange = 1e6, #Hz frequency for the implemented variation of nsb var rate

        gain_extraction_method = "baseline", #pulse, baseline
        
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
        self.ampDist_drift = 1.025457559561722
        self.max_nsb_var = max_nsb_var
        self.nsb_fchange = nsb_fchange
        self.gain_extraction_method = gain_extraction_method


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
            self.pulseShape = self.simulatePulseShape()
        elif type(pulseShape) is str:
            ps = np.loadtxt(pulseShape, unpack=True)
            # ensure correct sampling
            step = ps[0][1] - ps[0][0]
            ps, t = resample(ps[1], int(step / self.t_step * ps[1].shape[0]), ps[0])
            self.pulseShape = (t, ps)
            # offset from center (for plots only)
            self.plotOffset = (t[-1] + t[0]) / 2


            ##calculate the parameters of the pulse shape
            #self.singePE_area = integrate.quad(ps, t, initial=0)[-1]
            

            self.singePE_area = self.integrateSignal(self.pulseShape[0], self.pulseShape[1])
            #print(sum_)
        else:
            self.pulseShape = pulseShape


        self.pulseMean, self.pulseStddev = self.statsCalc(self.pulseShape[0], self.pulseShape[1])
        self.ampMean, self.ampStddev = self.statsCalc(self.ampSpec[0], self.ampSpec[1])

        # get distributions of spectra
        sx = self.timeSpec[0]
        bins = np.append(sx, sx[-1] + sx[1] - sx[0])
        self.timeDist = rv_histogram((self.timeSpec[1], bins))
        sx = self.ampSpec[0]
        bins = np.append(sx, sx[-1] + sx[1] - sx[0])
        self.ampDist = rv_histogram((self.ampSpec[1], bins))


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

        sum_ = 0
        for i in signal:
            sum_ += i*self.t_step # maybe wrong
        return sum_

    def statsCalc(self, times, signals):

        mean = signals.mean()
        stddev = np.std(signals-np.ones(signals.shape)*mean)
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

    def simulatePulseShape(self, t_mu=0, t_sig=3):
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
        psx = np.arange(int((6 * t_sig) / self.t_step)) * self.t_step - 3 * t_sig
        return psx, norm.pdf(psx, t_mu, t_sig)

    def simulatePMTSignal(self, peTimes, k_evts):
        """
        Simulates PMT signal based on photo electron times.

        Parameters
        ----------
        peTimes - array_like
                list of photo electron arrival times in ns

        Returns
        -------
        tuple of ndarray
                times in ns and simulated signal
        """

        

        # make discrete times
        t_min = peTimes.min() - self.t_pad
        t_max = peTimes.max() + self.t_pad
        times = np.arange((int(t_max - t_min) / self.t_step)) * self.t_step + t_min
        # make signal
        signal = np.zeros(times.shape)

        uncertainty = np.zeros(times.shape)

        #We need a way to know the number of Pe with same time
        #This information is in k_evts, need to be careful with indexes


        if self.background_rate_method == "poisson":

            for i, t in enumerate(peTimes):

                amp_rvs = self.ampDist.rvs() / self.ampDist_drift

                t += self.timeDist.rvs()
                signal[int((t - t_min) / self.t_step)] += amp_rvs
                uncertainty[int((t - t_min) / self.t_step)] = self.ampStddev * k_evts[i] + amp_rvs*math.sqrt(k_evts[i])
                
        return times, signal, uncertainty

    def simulateElectronics(self, signal, uncertainty):
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

        #We can convolve with a constant, see overleaf for equation
        uncertainty = np.convolve(signal, self.pulseStddev, "same") + np.convolve(uncertainty, self.pulseShape[1], "same")

        return np.convolve(signal, self.pulseShape[1], "same"), uncertainty

    def simulateADC(self, times, signal, uncertainty):
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

        samples = samples.astype(int)

        samples_unpro = signal[jitter :: self.oversamp]
        samples_unpro = samples_unpro.astype(int)

        """
        print("gain", self.gain)
        print("signal", signal[int(len(signal)/2)])
        print("sample", samples[int(len(samples)/2)])
        print("norm", norm.rvs(self.offset, self.noise, stimes.shape)[int(len(samples)/2)])
        """

        #Removing padding for easier readability
        if self.remove_padding:
            extra_padding = 5
            stimes = stimes[int(self.t_pad // 4)+extra_padding:]
            stimes = stimes[:len(stimes) - int(self.t_pad // 4) - extra_padding]
            samples = samples[int(self.t_pad // 4)+extra_padding:]
            samples = samples[:len(samples) - int(self.t_pad // 4) - extra_padding]

            samples_unpro = samples_unpro[int(self.t_pad // 4)+extra_padding:]
            samples_unpro = samples_unpro[:len(samples_unpro) - int(self.t_pad // 4) - extra_padding]

            uncertainty_sampled = uncertainty_sampled[int(self.t_pad // 4)+extra_padding:]
            uncertainty_sampled = uncertainty_sampled[:len(uncertainty_sampled) - int(self.t_pad // 4) - extra_padding]

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

        #calculate the poissonian lamda and exp mu

        lamda = self.t_step * self.background_rate * 1e-9#From the def E(poisson) = lamda

        mu = self.background_rate * 1e-9#From the def E(exp) = 1/mu

        #Implement change of nsb per second
        var_time = (1/self.nsb_fchange) * 1e9 #ns

        #first value of the background rate for reference
        bg_ref = self.background_rate

        #We define an array storing the uncertainty at every stime
        #uncertainty = []


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

                """
                print("times",time_delay)
                print("mu", mu)
                print("bck", self.background_rate)
                """
                
                if time_delay > var_time:
                    self.background_rate += norm.rvs(0, ((self.max_nsb_var * var_time * bg_ref)/(1e9))/(2*math.sqrt(2*math.log(2))), 1)[0]
                    mu = self.background_rate * 1e-9#From the def E(exp) = 1/mu
                    time_delay = 0
                    #print(((self.max_nsb_var * bg_ref)/(var_time ))/(2*math.sqrt(2*math.log(2))))

            evts = np.array(evts_list)


        elif self.background_rate_method == "poisson":



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

                for i, q in enumerate(poisson.rvs(abs(lamda), size=n_steps_var_time)):
                    # * operator create q times the same list
                    evts_list.extend([t_min + j*var_time + i * self.t_step] * q)
                    k_evts.extend([q] * q)
                    #So the time linked to i has q values
                    #We append to the uncertainty, index i (implicit)
                    #uncertainty.append(q*self.ampStddev+self.ampMean*math.sqrt(lamda)) #TODO : self.ampMean should depend on the actual value of ampdist.rvs, not constant

                #print(self.background_rate)
                self.background_rate += norm.rvs(0, ((self.max_nsb_var * var_time * bg_ref)/(1e9))/(2*math.sqrt(2*math.log(2))), 1)[0]
                lamda = self.t_step * self.background_rate * 1e-9#From the def E(poisson) = lamda

            #fill the last interval
            for i, q in enumerate(poisson.rvs(abs(lamda), size=int((t_max - n_intervals*var_time) // self.t_step))):
                evts_list.extend([t_min + var_time*n_intervals + i * self.t_step] * q)
                k_evts.extend([q] * q)

            evts = np.array(evts_list)
        
        return evts, k_evts

    def FPGA(self, times, signal, samples_unpro, uncert):

        bl = signal[0]
        bl_array = []

        for i in range(len(times)):
            if signal[i] > bl:
                bl += 0.125

            elif signal[i] < bl:
                bl -= 0.125

            bl_array.append(bl)

        bl_unpro = samples_unpro[0]
        bl_array_unpro = []

        for i in range(len(times)):
            if samples_unpro[i] > bl_unpro:
                bl_unpro += 0.125

            elif samples_unpro[i] < bl_unpro:
                bl_unpro -= 0.125

            bl_array_unpro.append(bl_unpro)

        #need to return both the baseline mean and signal mean

        #For now (needs to be improved), the uncertainty of bl mean is the average of uncertainties

        uncert_bl_mean = statistics.fmean(uncert)

        #print(uncert_bl_mean)

        bl_mean = statistics.fmean(bl_array)
        s_mean = statistics.fmean(signal)

        return bl_mean, s_mean, np.std(bl_array-signal), np.std(bl_array_unpro-samples_unpro), uncert_bl_mean


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
