#!/usr/bin/env python3

import scipy.integrate
from matplotlib import pyplot, rc
import numpy as np
from scipy.special import erfc
from numpy import exp, pi, sqrt, inf
from scipy.integrate import quad
import scipy.stats
from matplotlib import pyplot as plt


"""
# Numerical integration - not stable. :(
def enbw(l, s, subdivisions_inner=250, subdivisions_outer=50):
    def inner(x, f):
        return exponnorm(x, l, s) * exp(-2.0j * pi * x * f)

    def outer(w, window_scale=10):
        return quad(inner, -window_scale * s, 2 * window_scale * (1 / l + s), args=w, limit=subdivisions_inner)[0]

    return quad(outer, -1e3, 1e3, limit=subdivisions_outer)[0]
"""

def exponnorm(x, l, s):
    return exp(0.5 * l * (l * s * s - 2.0 * x)) * erfc((l * s * s - x) / (sqrt(2.0) * s))

def get_pulse_window(l, s, eps=1e-16):
    # Too lazy to code the isf myself...
    rvdist = scipy.stats.exponnorm(1.0 / (l * s), scale=s)
    a, b = rvdist.isf(1.0 - eps), rvdist.isf(eps)
    return a, b

def exponnorm_integral(l, s, eps=1e-16):
    # return quad(exponnorm, *get_pulse_window(l, s, eps=eps), args=(l, s))[0]
    return 2.0 / l

def enbw_dft(x : np.ndarray, dt : float):
    """Calculates the equivalent noise bandwidth of the signal `x` with
    sampling time `dt` using the discrete Fourier transform."""

    df = 1.0 / (dt * x.size)
    x = np.fft.rfft(x)

    return (np.abs(x)**2).sum() / np.abs(x[0])**2

def exponnorm_enbw(l, s, n=2048, eps=1e-16):
    t, dt = np.linspace(*get_pulse_window(l, s, eps=eps), n, retstep=True)
    x = exponnorm(t, l, s)
    return enbw_dft(x, dt)



def test_enbw():
    """
    Iterates through some 1st order low-pass configurations and
    compares the calculated ENBW with the analytical one from [1].

    [1]: https://analog.intgckts.com/equivalent-noise-bandwidth/
    """
    def filter(x : np.ndarray, dt : float, fc : float):
        """
        1st-order low-pass from [1]
        [1]: https://en.wikipedia.org/wiki/Low-pass_filter#Simple_infinite_impulse_response_filter
        """
        RC = 1.0 / (2.0 * np.pi * fc)
        alpha = dt / (RC + dt)

        y = x.copy()
        for i in range(1, y.size):
            y[i] = alpha * x[i] + (1.0 - alpha) * y[i - 1]

        return y

    for fc in np.logspace(-9, 9, 5):
        for dt in np.logspace(-5, -3, 5) / fc:
            tlim = (50 / (2 * np.pi * fc)) // dt * dt
            t = np.arange(-tlim, tlim, dt)
            x = np.zeros_like(t)
            x[x.size // 2] = 1.0
            y = filter(x, dt, fc)
            bw = enbw_dft(y, dt)
            print(bw / (0.5 * np.pi * fc))  # should be close to 1

# Parameters roughly describing the FlashCam 7-dynode pulse shape
l, s = 0.0659, 2.7118
print(f"Pulse integral: {exponnorm_integral(l, s):.3f}")
print(f"Equivalent noise bandwidth: {exponnorm_enbw(l, s) * 1e3:.1f} MHz")
