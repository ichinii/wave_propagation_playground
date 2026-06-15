#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import math
import timeit
import jax
import jax.numpy as jnp
import jax.lax as lax
import rayleigh_sommerfeld
import rayleigh_sommerfeld_batched

# jax.config.update("jax_transfer_guard", "log")
# jax.config.update("jax_transfer_guard", "log_explicit")

### scene ###

def create_empty_scene(name, samples_per_wavelength, wavelength):
    scene = {}
    scene["name"] = name
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

@jax.jit
def total_power(d_intensity, d_dx):
    return jnp.sum(d_intensity * d_dx)

def create_sequential_trace_dag(n):
    if n == 0: return []
    dag = [[]]
    for i in range(n-1):
        dag.append([i])
    return dag

def trace(algorithm, scene):
    print(f"tracing scene: \"{scene['name']}\" with {len(scene['objs'])} objects")

    # if there is no DAG preset. create one where each object depends on the previous one (sequential propagation)
    if not "trace_dag" in scene.keys():
        scene["trace_dag"] = create_sequential_trace_dag(len(scene["objs"]))

    d_scene = algorithm.instantiate(scene)
    d_fields = []

    for i, deps in enumerate(scene["trace_dag"]):
        if len(deps) == 0:
            d_fields.append(jnp.array([1.0 + 0.0j]))
        else:
            d_dst_fields = []
            for d in deps:
                d_dst_fields.append(algorithm.propagate(d_scene, d, i, d_fields[d]))
            d_fields.append(accumulate(d_dst_fields))

    d_intensities = [intensity(field) for field in d_fields]
    h_intensities = [np.array(jax.device_get(intensity)) for intensity in d_intensities]
    d_total_powers = [total_power(d_intensity, d_scene["objs"][i]["dx"]) for i, d_intensity in enumerate(d_intensities)]
    h_total_powers = [np.array(jax.device_get(d_total_power)) for d_total_power in d_total_powers]

    for i, deps in enumerate(scene["trace_dag"]):
        tot = h_total_powers[i]
        tot_deps = sum([h_total_powers[d] for d in deps])
        print(f"{deps} -> {i}:")
        print(f"  total power = {h_total_powers[i]}")
        print(f"  power factor = {tot/tot_deps if tot_deps > 0 else float('inf')}")

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
wavelength = 0.00123456789

def create_scene_law_of_reflection():
    scene = create_empty_scene("Law Of Reflection", samples_per_wavelength=samples_per_wavelength, wavelength=wavelength)
    # scene_append_point(scene, [-10, 10])
    scene_append_line(scene, [-16, 4], [-4, 16])
    scene_append_line(scene, [-0.1, 0], [0.1, 0])
    scene_append_line(scene, [8, 10], [12, 10])
    scene_append_line(scene, [18, 0], [22, 0])
    return scene

def create_scene_hard_cutoff():
    scene = create_empty_scene("Hard Cutoff", samples_per_wavelength=samples_per_wavelength, wavelength=wavelength)
    scene_append_point(scene, [-10, 0])
    scene_append_line(scene, [0, 0], [0, 5])
    scene_append_line(scene, [10, -5], [10, 5])
    return scene

def create_scene_single_slit():
    scene = create_empty_scene("Single Slit", samples_per_wavelength=samples_per_wavelength, wavelength=wavelength)
    slit_width = scene["wavelength"] * 32

    scene_append_line(scene, [-10, -10], [-10, 10])
    scene_append_line(scene, [0, slit_width / -2], [0, slit_width / 2])
    scene_append_line(scene, [10, -10], [10, 10])
    return scene

def create_scene_double_slit():
    scene = create_empty_scene("Double Slit", samples_per_wavelength=samples_per_wavelength, wavelength=wavelength)
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

def create_scene_sequential_beam(n):
    scene = create_empty_scene(f"Sequential Beam ({n})", samples_per_wavelength=samples_per_wavelength, wavelength=wavelength)
    for i in range(n):
        x = i/(n-1)*10
        scene_append_line(scene, [x, 0], [x, 10])
    return scene

scenes = [create_scene_law_of_reflection(), create_scene_hard_cutoff(), create_scene_single_slit(), create_scene_double_slit()]
# scenes = [create_scene_law_of_reflection()]
# scenes = [create_scene_sequential_beam(2)]
# scenes = [create_scene_sequential_beam(2), create_scene_sequential_beam(10)]

algorithm = rayleigh_sommerfeld

print()
print("WARMUP")
[trace(algorithm, scene) for scene in scenes] # warmup to trigger JIT compilation before timing

print()
print("GO")
def perf():
    return timeit.timeit(lambda: [trace(algorithm, scene) for scene in scenes] , number=1)
print(f"total time: {perf()} seconds")

# [plot(scene, trace(algorithm, scene)[0]) for scene in scenes] # warmup to trigger JIT compilation before timing

print(f"cache size: {algorithm._rayleigh_sommerfeld_kernel._cache_size()}")
