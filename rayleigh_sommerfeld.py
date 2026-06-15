import numpy as np
import math
import jax
import jax.numpy as jnp

def _instantiate_object(obj, samples_per_wavelength, wavelength):
    def instantiate_point(obj):
        return {
            "pos_x": jnp.array([obj["pos"][0]]),
            "pos_y": jnp.array([obj["pos"][1]]),
            "normal_x": jnp.array([1.0]),
            "normal_y": jnp.array([0.0]),
            "dx": jnp.array([1.0]),
        }

    def instantiate_line(obj):
        pos_a = np.array(obj["pos_a"])
        pos_b = np.array(obj["pos_b"])
        l = np.sqrt((pos_a[0] - pos_b[0])**2 + (pos_a[1] - pos_b[1])**2)
        n = math.ceil(samples_per_wavelength * l / wavelength)
        v = pos_b - pos_a
        return {
            "pos_x": jnp.linspace(pos_a[0], pos_b[0], n),
            "pos_y": jnp.linspace(pos_a[1], pos_b[1], n),
            "normal_x": jnp.array([v[1] / l]),
            "normal_y": jnp.array([v[0] / -l]),
            "dx": jnp.array([l / (n - 1)]),
        }

    if obj["type"] == "point": return instantiate_point(obj)
    if obj["type"] == "line": return instantiate_line(obj)
    else:
        raise ValueError(f"unknown obj type: {obj['type']}")

def instantiate(scene):
    d_scene = {
        "wavelength": jnp.array(scene["wavelength"]),
    }

    def instantiate_object(obj):
        return _instantiate_object(obj, scene["samples_per_wavelength"], scene["wavelength"])
    d_scene["objs"] = [instantiate_object(obj) for obj in scene["objs"]]

    return d_scene

"""
    2D Rayleigh-Sommerfeld diffraction integral (RS1)
    far-field approximation, because we are missing the Henkel function term that accounts for near-field effects

    @param src_pos_x: x coordinates of samples in the source object
    @param src_pos_y: y coordinates of samples in the source object
    @param src_normal_x: x component of the normal vector of the source object (for lines), or None for points
    @param src_normal_y: y component of the normal vector of the source object (for lines), or None for points
    @param src_dx: spacing between samples in the source object (for lines), or 1 for points
    @param src_field: complex amplitude of the wave at each sample in the source object
    @param dst_pos_x: x coordinates of samples in the destination object
    @param dst_pos_y: y coordinates of samples in the destination object
    @param wavelength: wavelength of the wave
"""
@jax.jit
def _rayleigh_sommerfeld_kernel(
    src_pos_x, src_pos_y, src_normal_x, src_normal_y, src_dx, src_field,
    dst_pos_x, dst_pos_y,
    wavelength
):
    # cartesian product of the samples in the source and destination objects
    # shapes: (num_dst, 1) and (1, num_src)
    ax = src_pos_x[jnp.newaxis, :]
    bx = dst_pos_x[:, jnp.newaxis]
    ay = src_pos_y[jnp.newaxis, :]
    by = dst_pos_y[:, jnp.newaxis]

    # 2. Distance and direction matrices
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
    cos_theta_line = jnp.abs(dir_x * src_normal_x[jnp.newaxis, :] + dir_y * src_normal_y[jnp.newaxis, :])

    # check if source is a single point using static shapes (evaluates at compile time)
    if src_pos_x.shape[0] == 1:
        cos_theta = jnp.ones_like(cos_theta_line)
    else:
        cos_theta = cos_theta_line

    # the contribution from each sample in the source to each sample in the destination is given by the Huygens-Fresnel principle
    k = 2.0 * jnp.pi / wavelength

    # Green's function mapping
    # shape: (num_dst, num_src)
    p = (jnp.exp(1j * k * l) / jnp.sqrt(l)) * cos_theta * src_field

    # riemann sum: sum over source samples of p * dx, where dx is the spacing between samples in the source object
    dst_field = jnp.sum(p, axis=1) * src_dx

    # normalization factor
    # ensures that the total power is conserved
    # note: 1/sqrt(1j*lambda) == sqrt(k/(2j*pi))
    norm_factor = 1.0 / jnp.sqrt(1j * wavelength)
    dst_field = norm_factor * dst_field

    return dst_field

def propagate(d_scene, ia, ib, d_src_field):
    return _rayleigh_sommerfeld_kernel(
        d_scene["objs"][ia]["pos_x"],
        d_scene["objs"][ia]["pos_y"],
        d_scene["objs"][ia]["normal_x"],
        d_scene["objs"][ia]["normal_y"],
        d_scene["objs"][ia]["dx"],
        d_src_field,
        d_scene["objs"][ib]["pos_x"],
        d_scene["objs"][ib]["pos_y"],
        d_scene["wavelength"],
    )
