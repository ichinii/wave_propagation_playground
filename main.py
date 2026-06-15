#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import math
import timeit
import jax
import jax.numpy as jnp
import jax.lax as lax
import rayleigh_sommerfeld

# jax.config.update("jax_transfer_guard", "log")
# jax.config.update("jax_transfer_guard", "log_explicit")

### scene ###

def create_empty_scene(samples_per_wavelength, wavelength):
    scene = {}
    scene["samples_per_wavelength"] = samples_per_wavelength
    scene["wavelength"] = wavelength
    scene["objs"] = []
    return scene

def scene_append_point(scene, pos):
    scene["objs"].append({
        "type": "point",
        "pos": pos,
    })

def scene_append_line(scene, pos_a, pos_b):
    scene["objs"].append({
        "type": "line",
        "pos_a": pos_a,
        "pos_b": pos_b,
    })

### simulate ###

@jax.jit
def accumulate(d_fields):
    stacked = jnp.stack(d_fields, axis=0)
    return jnp.sum(stacked, axis=0)

@jax.jit
def intensity(d_field):
    return jnp.abs(d_field) ** 2

def create_sequential_trace_dag(n):
    if n == 0: return []
    dag = [[]]
    for i in range(n-1):
        dag.append([i])
    return dag

def trace(algorithm, scene):
    # for i in range(scene_num_objects(scene)):
    #     n = object_num_samples(scene, i)
    #     l = object_length(scene, i)
    #     dx = object_dx(scene, i)
    #     print(f'{i}: n={n}, l={l}, dx={dx}')

    # if there is no DAG preset. create one where each object depends on the previous one (sequential propagation)
    if not "trace_dag" in scene.keys():
        scene["trace_dag"] = create_sequential_trace_dag(len(scene["objs"]))

    d_scene = algorithm.instantiate(scene)
    d_fields = []

    for i, deps in enumerate(scene["trace_dag"]):
        print(f"simulate object {i}. depends on {deps}")

        if len(deps) == 0:
            d_fields.append(jnp.array([1.0 + 0.0j]))
        else:
            d_dst_fields = []
            for d in deps:
                d_dst_fields.append(algorithm.propagate(d_scene, d, i, d_fields[d]))
            d_fields.append(accumulate(d_dst_fields))

    # [print(field) for field in d_fields]
    d_intensities = [intensity(field) for field in d_fields]
    h_intensities = [np.array(jax.device_get(intensity)) for intensity in d_intensities]
    print("done")
    return h_intensities

### analysis ###

def plot(scene, intensities):
    for i, intensity in enumerate(intensities):
        if len(intensity) == 1:
            plt.plot(intensity, label=f'{i}', marker='o')
        else:
            plt.plot(intensity, label=f'{i}')

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

samples_per_wavelength = 4
wavelength = 0.005123456789

def create_scene_law_of_reflection():
    scene = create_empty_scene(samples_per_wavelength=samples_per_wavelength, wavelength=wavelength)
    # scene_append_point(scene, [-10, 10])
    scene_append_line(scene, [-16, 4], [-4, 16])
    scene_append_line(scene, [-0.1, 0], [0.1, 0])
    scene_append_line(scene, [8, 10], [12, 10])
    scene_append_line(scene, [18, 0], [22, 0])
    return scene

def create_scene_hard_cutoff():
    scene = create_empty_scene(samples_per_wavelength=samples_per_wavelength, wavelength=wavelength)
    scene_append_point(scene, [-10, 0])
    scene_append_line(scene, [0, 0], [0, 5])
    scene_append_line(scene, [10, -5], [10, 5])
    return scene

def create_scene_single_slit():
    scene = create_empty_scene(samples_per_wavelength=samples_per_wavelength, wavelength=wavelength)
    slit_width = scene["wavelength"] * 32

    scene_append_line(scene, [-10, -10], [-10, 10])
    scene_append_line(scene, [0, slit_width / -2], [0, slit_width / 2])
    scene_append_line(scene, [10, -10], [10, 10])
    return scene

def create_scene_double_slit():
    scene = create_empty_scene(samples_per_wavelength=samples_per_wavelength, wavelength=wavelength)
    slit_width = scene["wavelength"] * 4
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
    scene = create_empty_scene(samples_per_wavelength=samples_per_wavelength, wavelength=wavelength)
    for i in range(4):
        scene_append_line(scene, [i, 0], [i, 10])
    return scene

# scene = create_scene_law_of_reflection()
# scene = create_scene_hard_cutoff()
# scene = create_scene_single_slit()
# scene = create_test_scene()
scenes = [create_scene_law_of_reflection(), create_scene_hard_cutoff(), create_scene_single_slit(), create_scene_double_slit(), create_test_scene()]
# scenes = [create_scene_double_slit()]

algorithm = rayleigh_sommerfeld
[trace(algorithm, scene) for scene in scenes] # warmup to trigger JIT compilation before timing
print(timeit.timeit(lambda:
    [trace(algorithm, scene) for scene in scenes]
, number=1))

# [plot(scene, trace(algorithm, scene)) for scene in scenes] # warmup to trigger JIT compilation before timing
