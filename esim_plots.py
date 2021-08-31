import numpy as np
from matplotlib import pyplot as plt
from electronics_simulation import ElectronicsSimulation

ampSpec = np.loadtxt("data/bb3_1700v_spe.txt",unpack=True)
timeSpec = "data/bb3_1700v_timing.txt"
pulseShape = np.loadtxt("data/bb3_1700v_pulse_shape.txt",unpack=True)

plt.plot(*ampSpec)
plt.xlabel("Amplitude")
plt.show()

# init class
esim = ElectronicsSimulation(ampSpec="data/bb3_1700v_spe.txt",timeSpec="data/bb3_1700v_timing.txt",pulseShape="data/bb3_1700v_pulse_shape.txt")
