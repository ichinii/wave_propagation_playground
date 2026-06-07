#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import math

n = 4000

def way1():
    sample_positions_x = [
        # np.full(n, -10.0),
        # np.linspace(-0.5, 0.5, n),
        # np.linspace(8.0, 12.0, n),

        np.full(n, -10.0),
        np.linspace(-0.5, 0.5, n),
        np.linspace(8.0, 12.0, n),
        np.linspace(-12, 20, n),
    ]

    sample_positions_y = [
        # np.full(n, -10.0),
        # np.full(n, 0.0),
        # np.full(n, 10.0),

        np.full(n, -10.0),
        np.full(n, 0.0),
        np.linspace(12.0, 8.0, n),
        np.full(n, 0.0),
    ]

    samples = np.full(n, 0.0)

    def ops(a, b):
        A, B = np.meshgrid(a, b, indexing='ij')
        return A.flatten(), B.flatten()

    def length(ax, ay, bx, by):
        return np.sqrt((ax - bx)**2 + (ay - by)**2)

    def f(l, w, s):
        return np.sin((2.0*math.pi/w) * l - s)# / np.sqrt(l)

    def F(p, R):
        return R * np.trapezoid(p.reshape(n, n), dx=1/n)

    def bounce(a, b, s, R):
        s = np.tile(samples, n)
        bx, ax = ops(sample_positions_x[b], sample_positions_x[a])
        by, ay = ops(sample_positions_y[b], sample_positions_y[a])
        l = length(ax, ay, bx, by)
        w = 0.001
        p = f(l, w, s)
        result = F(p, R)
        return result

    samples = bounce(0, 1, samples, 1)
    samples = bounce(1, 2, samples, 1)
    samples = bounce(2, 1, samples, 1)
    samples = bounce(2, 3, samples, 1)

    return samples

def way0():
    def l(x, p):
        return np.sqrt((p[0] - x)**2 + p[1]**2)

    def f(x, p0, p1, w, s):
        return (1 / np.sqrt(l(x, p0)*l(x, p1))) * np.sin((2*math.pi/w) * (l(x, p0)+l(x, p1)) + s)

    def F(p0, p1, m, w, s, R, I):
        x = np.linspace(-m/2, m/2, I)
        y = f(x, p0, p1, w, s)
        return R * np.trapezoid(y, dx=1/I)

    result = []
    I = n
    x = np.linspace(8, 12, I)
    for i in x:
        result.append(F([-10, 10], [i, 10], 1, 0.001, 0.2, 1, I))

    return result

# result = way0()
# plt.plot(np.linspace(8, 12, n), result)
# plt.show()
result = way1()
plt.plot(np.linspace(-12, 20, n), result)
plt.show()


# reference
## how many samples per wavelength

# monte carlo
## consider validity
## consider if sampling gains information, that can be used for importance sampling
## importance sampling: use attenuation distribution
