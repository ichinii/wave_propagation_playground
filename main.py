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

    def cartesian_product(a, b):
        A, B = np.meshgrid(a, b, indexing='ij')
        return A.flatten(), B.flatten()

    # propagate the samples from object ia to object ib, given the samples in the source object
    def propagate(ia, ib, s):
        # cartesian product of the samples in the source and destination objects
        bx, ax = cartesian_product(scene.pos_x[ib], scene.pos_x[ia])
        by, ay = cartesian_product(scene.pos_y[ib], scene.pos_y[ia])
        num_samples = len(ax)

        # the vector from each sample in the source to each sample in the destination
        v = np.array([bx - ax, by - ay]).T
        l = length(ax, ay, bx, by)
        d = v / l[:, np.newaxis]

        # RS1 obliquity factor: cos(theta) where theta is the angle between the normal of the source and the direction to the destination
        # this is Rayleigh-Sommerfeld correction to the Huygens-Fresnel principle,
        # which accounts for the fact that the contribution from each sample in the source is not isotropic,
        # but depends on the angle to the destination
        # ensures that the total power is conserved
        if (scene.normal[ia] is None):
            cos_theta = 1.0
        else:
            n = np.tile(scene.normal[ia], (num_samples, 1))
            cos_theta = np.abs(np.sum(d * n, axis=1))

        # repeat the source samples for each destination sample
        s = np.tile(s, object_num_samples(scene, ib))

        # the contribution from each sample in the source to each sample in the destination is given by the Huygens-Fresnel principle
        k = 2.0 * np.pi / scene.wavelength
        p = np.exp(1j * k * l) / np.sqrt(l) * cos_theta * s

        # reshape the result to have one row per destination sample and one column per source sample,
        # in order to sum the contributions from each source sample to each destination sample
        p = p.reshape(object_num_samples(scene, ib), object_num_samples(scene, ia))

        # riemann sum: sum over source samples of p * dx, where dx is the spacing between samples in the source object
        result = np.sum(p, axis=1) * object_dx(scene, ia)

        # normalization factor: 1/sqrt(i * wavelength)
        # ensures that the total power is conserved
        norm_factor = 1.0 / np.sqrt(1j * scene.wavelength)
        return norm_factor * result

    # if there is no DAG preset. create one where each object depends on the previous one (sequential propagation)
    if not hasattr(scene, "trace"):
        scene.trace = [ [] ]
        for i in range(scene_num_objects(scene) - 1):
            scene.trace.append([i])

    samples = []

    for i, deps in enumerate(scene.trace):
        print(f"{i} depends on {deps}")
        if len(deps) == 0:
            print(f"initial condition for {i}")
            s = np.full(object_num_samples(scene, i), cmath.rect(1, 0))
            samples.append(s)
        elif len(deps) == 1:
            print(f"propagate {deps[0]} -> {i}")
            s = propagate(deps[0], i, samples[deps[0]])
            samples.append(s)
        else:
            res = []
            for d in deps:
                print(f"propagate {d} -> {i}")
                res.append(propagate(d, i, samples[d]))
            print(f"sum contributions for {i}")
            res = np.sum(res, axis=0)
            samples.append(res)

    return samples

### analysis ###

def intensity(a):
    return np.abs(a)**2

def total_power(a, dx):
    return np.sum(intensity(a)) * dx

def plot(scene):
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

    plt.xlabel('sample index')
    plt.ylabel('intensity')
    fig = plt.gcf()
    plt.legend()
    plt.show()
    plt.draw()
    fig.savefig('img/prev.png')

### experiment ###

def create_scene_law_of_reflection():
    scene = create_empty_scene(samples_per_wavelength=4, wavelength=0.003123456789)
    # scene_append_point(scene, [-10, 10])
    scene_append_line(scene, [-16, 4], [-4, 16])
    scene_append_line(scene, [-0.1, 0], [0.1, 0])
    scene_append_line(scene, [8, 10], [12, 10])
    scene_append_line(scene, [18, 0], [22, 0])
    return scene

def create_scene_hard_cutoff():
    scene = create_empty_scene(samples_per_wavelength=4, wavelength=0.003123456789)
    scene_append_point(scene, [-10, 0])
    scene_append_line(scene, [0, 0], [0, 5])
    scene_append_line(scene, [10, -5], [10, 5])
    return scene

def create_scene_single_slit():
    scene = create_empty_scene(samples_per_wavelength=4, wavelength=0.003123456789)
    slit_width = scene.wavelength * 32

    scene_append_line(scene, [-10, -10], [-10, 10])
    scene_append_line(scene, [0, slit_width / -2], [0, slit_width / 2])
    scene_append_line(scene, [10, -10], [10, 10])
    return scene

def create_scene_double_slit():
    scene = create_empty_scene(samples_per_wavelength=4, wavelength=0.003123456789)
    slit_width = scene.wavelength * 8
    slit_spacing = slit_width * 4

    scene_append_line(scene, [-10, -10], [-10, 10])
    slit_radius = slit_width / 2
    slit_spacer = slit_spacing / 2 + slit_radius
    scene_append_line(scene, [0, -slit_radius - slit_spacer], [0, slit_radius - slit_spacer])
    scene_append_line(scene, [0, -slit_radius + slit_spacer], [0, slit_radius + slit_spacer])
    scene_append_line(scene, [10, -10], [10, 10])
    scene.trace = [
        [],
        [0],
        [0],
        [1, 2],
    ]
    return scene

plot(create_scene_law_of_reflection())
# plot(create_scene_hard_cutoff())
# plot(create_scene_single_slit())
# plot(create_scene_double_slit())
