#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm, rv_histogram, randint, poisson, expon
from scipy.signal import resample
from pulser import Pulser


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
        noise=0.8,

        gain=14,# in fadc_amplitude in CTA.cfg; ALSO : gain=7.5e5 from CTA-ULTRA5-small-4m-dc.cfg
        jitter=None,
        debugPlots=False,
        timeSpec=None,
        ampSpec=None,
        pulseShape=None,

        ##Flashcam parameters
        transit_time_jitter = 0.75, #From CTA.cfg, also : 0.64 Obtained from pmt_specs_R11920.cfg
        #electronic_noise_amplitude = 4.0, #From CTA.cfg, high gain

        background_rate_method = "exponential", #poisson, exponential
        background_rate = 100, #Hz
        show_graph = False
        
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
        #self.electronic_noise_amplitude = electronic_noise_amplitude
        self.background_rate_method = background_rate_method
        self.background_rate = background_rate
        self.show_graph = show_graph


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
            self.ampSpec = [np.loadtxt(ampSpec, unpack=True)[0],np.loadtxt(ampSpec, unpack=True)[1]]
            
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
        else:
            self.pulseShape = pulseShape
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

    def simulatePMTSignal(self, peTimes):
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
        for t in peTimes:
            t += self.timeDist.rvs()
            signal[int((t - t_min) / self.t_step)] += self.ampDist.rvs()  # TODO correct?
        return times, signal

    def simulateElectronics(self, signal):
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
        return np.convolve(signal, self.pulseShape[1], "same")

    def simulateADC(self, times, signal):
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
        samples = signal[jitter :: self.oversamp] * self.gain + norm.rvs(self.offset, self.noise, stimes.shape) #* self.electronic_noise_amplitude
        samples = samples.astype(int)
        return stimes, samples

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

        lamda = self.t_step * self.background_rate #From the def E(poisson) = lamda

        mu = self.background_rate #From the def E(exp) = 1/mu

        print(mu)
        print(lamda)

        
        if self.background_rate_method == "exponential":
            if len(evts) > 0:
                t_min = evts.min() - self.t_pad
                t_max = evts.max() + self.t_pad
                evts_list = evts.tolist()
            else :
                t_min = (-1) * self.t_pad
                t_max = self.t_pad
                evts_list = []
            sxap = t_min

            #convert np.array() to python list
            
            while sxap < t_max:
                sxap += expon.rvs(scale = mu)
                evts_list.append(sxap)

            evts = np.array(evts_list)
        elif self.background_rate_method == "poisson":
            if len(evts) > 0:
                t_min = evts.min() - self.t_pad
                t_max = evts.max() + self.t_pad
                #convert np.array() to python list
                evts_list = evts.tolist()
            else :
                t_min = (-1) * self.t_pad
                t_max = self.t_pad
                evts_list = []

            

            for i, q in enumerate(poisson.rvs(self.br_mu, size=(t_max - t_min) // t_step)):
                evts_list.extend([t_min + i * t_step] * q)

            evts = np.array(evts_list)
        
        return evts


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

    #ampSpec = np.loadtxt("data/bb3_1700v_spe.txt", unpack=True)

    #Amplitude spectrum obtained from spe_R11920-RM_ap0.0002.dat
    ampSpec = np.loadtxt("data/bb3_1700v_spe.txt", unpack=True)
    timeSpec = "data/bb3_1700v_timing.txt"
    pulseShape = np.loadtxt("data/bb3_1700v_pulse_shape.txt", unpack=True)

    # init class
    esim = TraceSimulation(
        ampSpec="data/spe_R11920-RM_ap0.0002.dat",
        timeSpec="data/bb3_1700v_timing.txt",
        pulseShape="data/bb3_1700v_pulse_shape.txt",
    )



    # plot spectra

    if esim.show_graph:
        plt.figure()
        plt.title("Amplitude spectrum")
        plt.plot(*esim.ampSpec)
        plt.xlabel("Amplitude")

        plt.figure()
        plt.title("Time spectrum")
        plt.plot(*esim.timeSpec)
        plt.xlabel("Time/ns")

        plt.figure()
        plt.title("Pulse shape")
        plt.plot(*esim.pulseShape)
        plt.xlabel("Time/ns")
        plt.ylabel("Amplitude")

    #Create a pulser class
    pulse = Pulser(step=esim.t_step)

    evts = pulse.generate_all()

    #we need to add random evts that follow a negative exponential for the background rate
    evts_br = esim.simulateBackground(evts)

    # pmt signal
    times, pmtSig = esim.simulatePMTSignal(evts_br)
    eleSig = esim.simulateElectronics(pmtSig)
    plt.figure()
    plt.title("Simulated signal")
    plt.scatter(evts_br, np.zeros(evts_br.shape))
    plt.bar(times, pmtSig)
    plt.plot(times + esim.plotOffset, eleSig)
    plt.xlabel("Time/ns")
    plt.ylabel("Amplitude")

    # adc signal
    stimes, samples = esim.simulateADC(times, eleSig)
    plt.figure()
    plt.title("Simulated ADC output")
    plt.plot(stimes + esim.plotOffset, samples)
    plt.xlabel("Time/ns")
    plt.ylabel("ADC output/LSB")


    plt.show()

    




if __name__ == "__main__":
    example_usage()
