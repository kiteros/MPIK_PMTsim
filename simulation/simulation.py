#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm, rv_histogram, randint, poisson, expon
from scipy.signal import resample

import statistics
import scipy.integrate as integrate
import math 
from calculate_gains import GainCalculator
from trace_simulation import TraceSimulation

gc = GainCalculator()
gc.extract_gain()