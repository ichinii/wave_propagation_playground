#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import math
import timeit
# import equinox as eqx
import jax
import jax.numpy as jnp
from jax import jit

# jax.config.update("jax_transfer_guard", "log")
# jax.config.update("jax_transfer_guard", "log_explicit")

### misc ###

def length(ax, ay, bx, by):
    return np.sqrt((ax - bx)**2 + (ay - by)**2)

def intensity(a):
    return np.abs(a)**2

def total_power(a, dx):
    return np.sum(intensity(a)) * dx

### scene ###

# TODO: prevent recompilation on different scenes
def create_empty_scene(samples_per_wavelength, wavelength):
    scene = {}
    scene["samples_per_wavelength"] = samples_per_wavelength
    scene["wavelength"] = wavelength
    scene["pos_x"] = [] # x coordinates of samples in each object
    scene["pos_y"] = [] # y coordinates of samples in each object
    scene["normal_x"] = [] # x component of the normal vector of each object
    scene["normal_y"] = [] # y component of the normal vector of each object
    scene["dx"] = [] # spacing between samples in each object (for lines), or 1 for points
    scene["num"] = [] # number of samples in each object
    return scene

def scene_append_point(scene, pos):
    scene["pos_x"].append(jnp.array([pos[0]]))
    scene["pos_y"].append(jnp.array([pos[1]]))
    scene["normal_x"].append(jnp.array(1.0))
    scene["normal_y"].append(jnp.array(0.0))
    scene["dx"].append(jnp.array(1.0))
    scene["num"].append(jnp.array(1))

def scene_append_line(scene, pos_a, pos_b):
    pos_a = np.array(pos_a)
    pos_b = np.array(pos_b)
    l = length(pos_a[0], pos_a[1], pos_b[0], pos_b[1])
    n = math.ceil(scene["samples_per_wavelength"] * l / scene["wavelength"])
    d = pos_b - pos_a
    scene["pos_x"].append(jnp.linspace(pos_a[0], pos_b[0], n))
    scene["pos_y"].append(jnp.linspace(pos_a[1], pos_b[1], n))
    scene["normal_x"].append(jnp.array(d[1] / l))
    scene["normal_y"].append(jnp.array(d[0] / -l))
    scene["dx"].append(jnp.array(l / (n - 1)))
    scene["num"].append(n)

def scene_num_objects(scene):
    return len(scene["pos_x"])

### simulate ###

"""
    2D Rayleigh-Sommerfeld diffraction integral (RS1)
    far-field approximation, because we are missing the Henkel function term that accounts for near-field effects

    @param src_pos_x: x coordinates of samples in the source object
    @param src_pos_y: y coordinates of samples in the source object
    @param src_normal_x: x component of the normal vector of the source object (for lines), or None for points
    @param src_normal_y: y component of the normal vector of the source object (for lines), or None for points
    @param src_dx: spacing between samples in the source object (for lines), or 1 for points
    @param src_field: complex amplitude of the wave at each sample in the source object
    @param src_num: number of samples in the source object
    @param dst_pos_x: x coordinates of samples in the destination object
    @param dst_pos_y: y coordinates of samples in the destination object
    @param dst_num: number of samples in the destination object
    @param wavelength: wavelength of the wave
"""
@jit
def rayleigh_sommerfeld_kernel(src_pos_x, src_pos_y, src_normal_x, src_normal_y, src_dx, src_field, src_num, dst_pos_x, dst_pos_y, dst_num, wavelength):
    # cartesian product of the samples in the source and destination objects
    ax = src_pos_x[jnp.newaxis, :]
    bx = dst_pos_x[:, jnp.newaxis]
    ay = src_pos_y[jnp.newaxis, :]
    by = dst_pos_y[:, jnp.newaxis]

    vel_x = bx - ax
    vel_y = by - ay
    l = jnp.sqrt(vel_x**2 + vel_y**2 + 1e-12)
    dir_x = vel_x / l
    dir_y = vel_y / l

    # RS1 obliquity factor: cos(theta) where theta is the angle between the normal of the source and the direction to the destination
    # this is Rayleigh-Sommerfeld correction to the Huygens-Fresnel principle,
    # which accounts for the fact that the contribution from each sample in the source is not isotropic,
    # but depends on the angle to the destination
    # ensures that the total power is conserved
    src_normal_x = jnp.atleast_1d(src_normal_x)
    src_normal_y = jnp.atleast_1d(src_normal_y)
    cos_theta_line = jnp.abs(dir_x * src_normal_x[jnp.newaxis, :] + dir_y * src_normal_y[jnp.newaxis, :])
    src_is_point = src_num == 1
    cos_theta = jnp.where(src_is_point, jnp.ones_like(cos_theta_line), cos_theta_line)

    # padding index masks to safeguard the Riemann Sum
    src_mask = jnp.arange(src_pos_x.shape[0]) < src_num
    dst_mask = jnp.arange(dst_pos_x.shape[0]) < dst_num
    grid_mask = dst_mask[:, jnp.newaxis] & src_mask[jnp.newaxis, :]

    # the contribution from each sample in the source to each sample in the destination is given by the Huygens-Fresnel principle
    k = 2.0 * jnp.pi / wavelength

    # p shape: (num_dest, num_src)
    p = (jnp.exp(1j * k * l) / jnp.sqrt(l)) * cos_theta * src_field[jnp.newaxis, :]

    # apply the padding masks to zero out contributions from invalid samples
    p = jnp.where(grid_mask, p, 0.0j)

    # riemann sum: sum over source samples of p * dx, where dx is the spacing between samples in the source object
    dst_field = jnp.sum(p, axis=1) * src_dx

    # normalization factor
    # ensures that the total power is conserved
    # note: 1/sqrt(1j*lambda) == sqrt(k/(2j*pi))
    norm_factor = 1.0 / jnp.sqrt(1j * wavelength)
    dst_field = norm_factor * dst_field

    # apply the destination mask to zero out contributions to invalid destination samples
    dst_field = jnp.where(dst_mask, dst_field, 0.0j)

    return dst_field

# @jit
# def intensity_kernel(field):
#     return jnp.abs(field)**2
#
# def intensity(field):
#     return intensity_kernel(field)
#
# @jit
# def total_power_kernel(intensity, src_dx):
#     return jnp.sum(intensity) * dx
#
# def total_power(intensity, src_dx):
#     return total_power_kernel(intensity, src_dx)

def rayleigh_sommerfeld(scene, ia, ib, s):
    return rayleigh_sommerfeld_kernel(
        scene["pos_x"][ia],
        scene["pos_y"][ia],
        scene["normal_x"][ia],
        scene["normal_y"][ia],
        scene["dx"][ia],
        s,
        scene["num"][ia],
        scene["pos_x"][ib],
        scene["pos_y"][ib],
        scene["num"][ib],
        jnp.array(scene["wavelength"])
    )

def trace(propagate, scene):
    # for i in range(scene_num_objects(scene)):
    #     n = object_num_samples(scene, i)
    #     l = object_length(scene, i)
    #     dx = object_dx(scene, i)
    #     print(f'{i}: n={n}, l={l}, dx={dx}')

    # if there is no DAG preset. create one where each object depends on the previous one (sequential propagation)
    if not hasattr(scene, "trace_dag"):
        scene["trace_dag"] = [ [] ]
        for i in range(scene_num_objects(scene) - 1):
            scene["trace_dag"].append([i])

    samples = []

    for i, deps in enumerate(scene["trace_dag"]):
        print(f"{i} depends on {deps}")

        if len(deps) == 0:
            print(f"initial condition for {i}")
            s = jnp.array([1.0 + 0.0j])
            samples.append(s)
        elif len(deps) == 1:
            print(f"propagate {deps[0]} -> {i}")
            s = propagate(scene, deps[0], i, samples[deps[0]])
            samples.append(s)
        else:
            res = []
            for d in deps:
                print(f"propagate {d} -> {i}")
                res.append(propagate(scene, d, i, samples[d]))
            print(f"sum contributions for {i}")
            res = np.sum(res, axis=0)
            samples.append(res)

    return [jax.device_get(s) for s in samples]

### analysis ###

def plot(scene, result):
    intens = [intensity(s) for s in result]
    tot = [total_power(s, jnp.atleast_1d(scene["dx"][i])[0]) for i, s in enumerate(result)]
    tot_factor = [tot[i+1] / tot[i] for i in range(len(result) - 1)]

    for i in range(len(result)):
        print(f'{i}: total power={tot[i]}')
        if len(result[i]) == 1:
            plt.plot(intens[i], label=f'{i}', marker='o')
        else:
            plt.plot(intens[i], label=f'{i}')

    # for i in range(len(tot_factor)):
    #     print(f'{i} -> {i+1}: power factor={tot_factor[i]}')

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
    scene = create_empty_scene(samples_per_wavelength=4, wavelength=0.00123456789)
    scene_append_point(scene, [-10, 0])
    scene_append_line(scene, [0, 0], [0, 5])
    scene_append_line(scene, [10, -5], [10, 5])
    return scene

def create_scene_single_slit():
    scene = create_empty_scene(samples_per_wavelength=4, wavelength=0.00123456789)
    slit_width = scene["wavelength"] * 32

    scene_append_line(scene, [-10, -10], [-10, 10])
    scene_append_line(scene, [0, slit_width / -2], [0, slit_width / 2])
    scene_append_line(scene, [10, -10], [10, 10])
    return scene

def create_scene_double_slit():
    scene = create_empty_scene(samples_per_wavelength=4, wavelength=0.00123456789)
    slit_width = scene["wavelength"] * 8
    slit_spacing = slit_width * 4

    scene_append_line(scene, [-10, -10], [-10, 10])
    slit_radius = slit_width / 2
    slit_spacer = slit_spacing / 2 + slit_radius
    scene_append_line(scene, [0, -slit_radius - slit_spacer], [0, slit_radius - slit_spacer])
    scene_append_line(scene, [0, -slit_radius + slit_spacer], [0, slit_radius + slit_spacer])
    scene_append_line(scene, [10, -10], [10, 10])
    scene["trace_dag"] = [
        [],
        [0],
        [0],
        [1, 2],
    ]
    return scene

def create_test_scene():
    scene = create_empty_scene(samples_per_wavelength=4, wavelength=0.005123456789)
    for i in range(4):
        scene_append_line(scene, [i, 0], [i, 10])
    return scene

# scene = create_scene_law_of_reflection()
# scene = create_scene_hard_cutoff()
# scene = create_scene_single_slit()
# scene = create_test_scene()
scenes = [create_scene_law_of_reflection(), create_scene_hard_cutoff(), create_scene_single_slit(), create_scene_double_slit(), create_test_scene()]

trace(rayleigh_sommerfeld, create_test_scene()) # warmup to trigger JIT compilation before timing
print(timeit.timeit(lambda:
    # [trace(rayleigh_sommerfeld, scene) for scene in scenes]
    trace(rayleigh_sommerfeld, create_test_scene())
, number=1))
print(rayleigh_sommerfeld_kernel._cache_size())

# plot(scene, trace(rayleigh_sommerfeld, scene))

