#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import rv_histogram, randint, poisson, expon, exponnorm
from scipy.signal import resample
import scipy.integrate as integrate
import scipy
import math 
from scipy.optimize import curve_fit
from scipy import odr
from pylab import *
import statistics
import os.path

from scipy.stats import norm

from mpl_toolkits.mplot3d import axes3d, Axes3D
from sklearn import linear_model 

import csv
import scipy.fftpack

from scipy import special




####lets put everything in a cute function

def get_baseline_uncertainty(N, sigma_PE):
	#N: number of samples
	#sigma_PE : uncertainty of one point

	####Array for les P+,i car P-,i=1-P+,i
	n = N
	P_plus = 0.5*np.ones(n+1)
	baseline_speed = 0.125

	##now P_plus depends on k
	for i in range(len(P_plus)):
		P_plus[i] = 1-scipy.stats.norm(0, sigma_PE).cdf(i*baseline_speed)




	nb_k = 2*n-1

	S = np.zeros((nb_k,n))
	middle_index = int((nb_k-1)/2)
	S[middle_index][0] = 1


	###calculate the standard deviation at every n
	standard_deviation = []
	standard_deviation.append(0) ###for n=0


	for i in range(n-1):
		i = i+1 ##on saute l'index 0


		###So here we are at k = 1
		#we have k+1 nodes, starting at k-1 from the previous one, incrementing by two
		for k in range(i+1):
			current_k = middle_index-i+2*k

			absolute_k = abs(current_k-middle_index)
			

			if k == 0:
				####premier element de la ligne
				##sera toujours en absolute k negatif
				S[current_k][i] = S[current_k+1][i-1]*P_plus[absolute_k-1]
			elif k == i:
				##dernier element de la ligne
				S[current_k][i] = S[current_k-1][i-1]*P_plus[absolute_k-1]
			else :

				###Ligne dapres only pour ceux du milieu
				if current_k-middle_index>0:

					S[current_k][i] = S[current_k+1][i-1]*(1-P_plus[absolute_k+1])+S[current_k-1][i-1]*P_plus[absolute_k-1]
				elif current_k-middle_index == 0:
					S[current_k][i] = S[current_k+1][i-1]*(1-P_plus[1])+S[current_k-1][i-1]*(1-P_plus[1])
				else:
					S[current_k][i] = S[current_k+1][i-1]*(P_plus[absolute_k-1])+S[current_k-1][i-1]*(1-P_plus[absolute_k+1])

			this_row = [x[i] for x in S]


		#we need to know the value of the 
		esp_x = 0
		esp_x2 = 0
		for k in range(i+1):
			current_k = middle_index-i+2*k
			non_absolute_k = current_k-middle_index
			esp_x += this_row[current_k]*non_absolute_k*baseline_speed
			esp_x2 += this_row[current_k]*(non_absolute_k*baseline_speed)**2

		std = np.sqrt(esp_x2-esp_x**2)
		
		standard_deviation.append(std)

		print("calculating row...", i)
		



	"""
	last_row = [x[-1] for x in S]


	###remove the 0
	last_row = [i for i in last_row if i != 0]

	plt.figure()
	plt.plot(last_row)
	plt.show()
	"""

	standard_deviation_baseline_mean = 0
	for x in standard_deviation:
		standard_deviation_baseline_mean += x**2

	standard_deviation_baseline_mean = (1/n)*np.sqrt(standard_deviation_baseline_mean)

	std_fixed = (1/np.sqrt(n))*standard_deviation[-1]
	return standard_deviation_baseline_mean, std_fixed

print(get_baseline_uncertainty(40, 3))