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
import os.path


class Baseline:


	def execute(esim_init, gain_min, gain_max):
		pulse = Pulser(step=esim_init.t_step, pulse_type="none")
		evts = pulse.generate_all()


		gains = np.arange(gain_min, gain_max + 1, 1)
		fig, ax = plt.subplots()
		ax.set_title("Th vs Exp gain")
		ax.plot(gains, gains)


		gain_2 = []
		noises = []
		exp_gain_unc = []


		for i in np.linspace(0.5, 0.5, num=nb_lines):

			print(i)
			noises.append(i)

			# training
			coeff, coeff_uncertainty, offset_coeff = esim_init.calculate_coeff(evts=evts, noi=i)

			# experience
			gains, slopes, slopes_uncertainties = self.loop_gain_bl(
		        evts=evts,
		        gain_min=self.gain_min,
		        gain_max=self.gain_max,
		        n1=self.n_exp_1,
		        n2=self.n_exp_2,
		        bk_min=self.bk_min,
		        bk_max=self.bk_max,
		        nsb_var=self.nsb_var,
		        noise_=i,
		    )

		    # gains, slopes, slopes_uncertainties = self.loop_gain_bl(evts=evts, gain_min=2, gain_max=2,
			# n1=self.n_exp_1, n2=self.n_exp_2, bk_min=self.bk_min, bk_max=self.bk_max, nsb_var=self.nsb_var, noise_=i)

			exp_gain = []
			exp_gain_unc = []

			for i, q in enumerate(slopes):
				unc_s = slopes_uncertainties[i]

				exp_gain.append(q / coeff)
				exp_gain_unc.append(unc_s / coeff + (q / (coeff * coeff)) * coeff_uncertainty)
			ax.plot(gains, exp_gain - offset_coeff, label="exp")

			# plt.errorbar(gains, exp_gain, yerr=exp_gain_unc, label=i)
			# print(exp_gain)

			# gain_2.append(exp_gain[0])

		ax.fill_between(gains, [a - b for a, b in zip(gains, exp_gain_unc)],[a + b for a, b in zip(gains, exp_gain_unc)],alpha=0.2)
		ax.legend(loc="upper left")

		# plt.figure()
		# plt.plot(noises, gain_2)

		plt.show()

		return 1


