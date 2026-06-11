#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import math
import cmath

### misc ###

def length(ax, ay, bx, by):
    return np.sqrt((ax - bx)**2 + (ay - by)**2)

### scene ###

def create_empty_scene(samples_per_wavelength, wavelength):
    scene = type('Scene', (), {})()
    scene.samples_per_wavelength = samples_per_wavelength
    scene.wavelength = wavelength
    scene.pos_x = []
    scene.pos_y = []
    scene.normal = []
    return scene

def scene_append_point(scene, pos):
    scene.pos_x.append(np.array([pos[0]]))
    scene.pos_y.append(np.array([pos[1]]))
    scene.normal.append(None)

def scene_append_line(scene, pos_a, pos_b):
    pos_a = np.array(pos_a)
    pos_b = np.array(pos_b)
    l = length(pos_a[0], pos_a[1], pos_b[0], pos_b[1])
    n = math.ceil(scene.samples_per_wavelength * l / scene.wavelength)
    scene.pos_x.append(np.linspace(pos_a[0], pos_b[0], n))
    scene.pos_y.append(np.linspace(pos_a[1], pos_b[1], n))
    d = pos_b - pos_a
    scene.normal.append([d[1] / l, d[0] / -l])

def scene_num_objects(scene):
    return len(scene.pos_x)

def object_num_samples(scene, i):
    return len(scene.pos_x[i])

def object_length(scene, i):
    if object_num_samples(scene, i) == 1:
        return 0
    return length(scene.pos_x[i][0], scene.pos_y[i][0], scene.pos_x[i][-1], scene.pos_y[i][-1])

def object_dx(scene, i):
    if object_num_samples(scene, i) == 1:
        return 1.0
    return object_length(scene, i) / (object_num_samples(scene, i) - 1)

### simulate ###

def huygens_fresnel(scene):
    for i in range(scene_num_objects(scene)):
        n = object_num_samples(scene, i)
        l = object_length(scene, i)
        dx = object_dx(scene, i)
        print(f'{i}: n={n}, l={l}, dx={dx}')

    s = np.full(object_num_samples(scene, 0), cmath.rect(1, 0))

    def cartesian_product(a, b):
        A, B = np.meshgrid(a, b, indexing='ij')
        return A.flatten(), B.flatten()

    def propagate(ia, ib, s):
        bx, ax = cartesian_product(scene.pos_x[ib], scene.pos_x[ia])
        by, ay = cartesian_product(scene.pos_y[ib], scene.pos_y[ia])
        num_samples = len(ax)

        v = np.array([bx - ax, by - ay]).T
        l = length(ax, ay, bx, by)
        d = v / l[:, np.newaxis]

        k = 2.0 * np.pi / scene.wavelength

        # if (scene.normal[ia] is None):
        #     cos_theta = 1.0
        # else:
        # TODO: ia or ib
        n = np.tile(scene.normal[ib], (num_samples, 1))
        cos_theta = np.abs(np.sum(d * n, axis=1))

        s = np.tile(s, object_num_samples(scene, ib))
        p = (np.exp(1j * k * l) / np.sqrt(l)) * cos_theta * s
        p = p.reshape(object_num_samples(scene, ib), object_num_samples(scene, ia))

        norm_factor = 1.0 / np.sqrt(1j * scene.wavelength)
        return norm_factor * np.sum(p, axis=1) * object_dx(scene, ia)

    samples = []
    samples.append(s)
    for i in range(scene_num_objects(scene) - 1):
        s = propagate(i, i + 1, s)
        samples.append(s)
    return samples

### analysis ###

def intensity(a):
    return np.abs(a)**2

def total_power(a, dx):
    return np.sum(intensity(a)) * dx

### experiment ###

scene = create_empty_scene(samples_per_wavelength=4, wavelength=0.00123456789)
# scene_append_point(scene, [-10, 10])
scene_append_line(scene, [-16, 4], [-4, 16])
scene_append_line(scene, [-0.1, 0], [0.1, 0])
scene_append_line(scene, [8, 10], [12, 10])
scene_append_line(scene, [18, 0], [22, 0])

result = huygens_fresnel(scene)
intens = [intensity(s) for s in result]
tot = [total_power(s, object_dx(scene, i)) for i, s in enumerate(result)]
tot_factor = [tot[i+1] / tot[i] for i in range(len(result) - 1)]

for i in range(len(result)):
    print(f'{i}: total power={tot[i]}')
    if len(result[i]) == 1:
        plt.plot(intens[i], label=f'{i}', marker='o')
    else:
        plt.plot(intens[i], label=f'{i}')

for i in range(len(tot_factor)):
    print(f'{i} -> {i+1}: power factor={tot_factor[i]}')

fig = plt.gcf()
plt.legend()
plt.show()
plt.draw()
fig.savefig('prev.png')


# reference
## how many samples per wavelength

# monte carlo
## consider validity
## consider if sampling gains information, that can be used for importance sampling
## importance sampling: use attenuation distribution
