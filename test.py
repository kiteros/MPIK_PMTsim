import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm, rv_histogram, randint, poisson, expon
from scipy.signal import resample
from pulser import Pulser


print(expon.rvs(scale = 1e2))