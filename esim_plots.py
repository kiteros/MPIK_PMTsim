import numpy as np
from matplotlib import pyplot as plt
from electronics_simulation import ElectronicsSimulation

ampSpec = np.loadtxt("data/bb3_1700v_spe.txt",unpack=True)
timeSpec = "data/bb3_1700v_timing.txt"
pulseShape = np.loadtxt("data/bb3_1700v_pulse_shape.txt",unpack=True)

# init class
esim = ElectronicsSimulation(ampSpec="data/bb3_1700v_spe.txt",timeSpec="data/bb3_1700v_timing.txt",pulseShape="data/bb3_1700v_pulse_shape.txt")

# plot spectra
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

# random input data
evts = np.random.default_rng().random(10)*100

# pmt signal
times, pmtSig = esim.simulatePMTSignal(evts)
eleSig = esim.simulateElectronics(pmtSig)
plt.figure()
plt.title("Simulated signal")
plt.scatter(evts,np.zeros(evts.shape))
plt.bar(times, pmtSig)
plt.plot(times+esim.plotOffset, eleSig)
plt.xlabel("Time/ns")
plt.ylabel("Amplitude")

# adc signal
stimes, samples = esim.simulateADC(times, eleSig)
plt.figure()
plt.title("Simulated ADC output")
plt.plot(stimes+esim.plotOffset, samples)
plt.xlabel("Time/ns")
plt.ylabel("ADC output/LSB")

plt.show()
