import numpy as np
import math
import jax
import jax.numpy as jnp
from jax import lax

# --- GLOBAL STATIC COMPILER HINTS ---
# Adjust these values to fit your targeted hardware constraints.
# Changing these values will trigger a clean, one-time re-compilation.
_SRC_CAPACITY = 2**14  # Maximum number of source samples allowed
_DST_CAPACITY = 2**14  # Maximum number of destination samples allowed
_BATCH_SIZE = 128     # Must divide cleanly into _DST_CAPACITY (e.g., 4096 // 128 = 32 batches)
assert _DST_CAPACITY % _BATCH_SIZE == 0, f"_DST_CAPACITY ({_DST_CAPACITY}) must be a clean multiple of _BATCH_SIZE ({_BATCH_SIZE})"

def _instantiate_object(obj, samples_per_wavelength, wavelength):
    """
    Allocates static, fixed-capacity buffers on the device.
    Fills up to the required sample count 'n', and pads the rest with zeros.
    """
    # 1. Determine true physical sample counts
    if obj["type"] == "point":
        n = 1
    elif obj["type"] == "line":
        pos_a = np.array(obj["pos_a"])
        pos_b = np.array(obj["pos_b"])
        l = np.sqrt((pos_a[0] - pos_b[0])**2 + (pos_a[1] - pos_b[1])**2)
        n = math.ceil(samples_per_wavelength * l / wavelength)
    else:
        raise ValueError(f"unknown obj type: {obj['type']}")

    # Hard boundary assertions to prevent silent memory truncations
    assert n <= _SRC_CAPACITY, f"Requested object samples ({n}) exceeds _SRC_CAPACITY ({_SRC_CAPACITY})"
    assert n <= _DST_CAPACITY, f"Requested object samples ({n}) exceeds _DST_CAPACITY ({_DST_CAPACITY})"

    # 2. Compute true physical parameters
    if obj["type"] == "point":
        raw_x = np.array([obj["pos"][0]], dtype=np.float32)
        raw_y = np.array([obj["pos"][1]], dtype=np.float32)
        raw_nx = np.array([1.0], dtype=np.float32)
        raw_ny = np.array([0.0], dtype=np.float32)
        raw_dx = np.array(1.0, dtype=np.float32) # Clean 0D scalar shape ()
    else:
        raw_x = np.linspace(pos_a[0], pos_b[0], n, dtype=np.float32)
        raw_y = np.linspace(pos_a[1], pos_b[1], n, dtype=np.float32)
        v = pos_b - pos_a
        raw_nx = np.linspace(v[1] / l, v[1] / l, n, dtype=np.float32)
        raw_ny = np.linspace(v[0] / -l, v[0] / -l, n, dtype=np.float32)
        raw_dx = np.array(l / (n - 1), dtype=np.float32) # Clean 0D scalar shape ()

    # 3. Create host-side zero padding structures up to global static capacities
    pad_src_x = np.zeros((_SRC_CAPACITY,), dtype=np.float32)
    pad_src_y = np.zeros((_SRC_CAPACITY,), dtype=np.float32)
    pad_src_nx = np.zeros((_SRC_CAPACITY,), dtype=np.float32)
    pad_src_ny = np.zeros((_SRC_CAPACITY,), dtype=np.float32)

    pad_dst_x = np.zeros((_DST_CAPACITY,), dtype=np.float32)
    pad_dst_y = np.zeros((_DST_CAPACITY,), dtype=np.float32)

    # 4. Splice physical measurements into the gates of the arrays
    pad_src_x[:n] = raw_x
    pad_src_y[:n] = raw_y
    pad_src_nx[:n] = raw_nx
    pad_src_ny[:n] = raw_ny

    pad_dst_x[:n] = raw_x
    pad_dst_y[:n] = raw_y

    # Return unified fixed-capacity device dictionaries
    return {
        "pos_x": jnp.array(pad_dst_x),      # Serves as destination array
        "pos_y": jnp.array(pad_dst_y),
        "src_pos_x": jnp.array(pad_src_x),  # Serves as source array
        "src_pos_y": jnp.array(pad_src_y),
        "normal_x": jnp.array(pad_src_nx),
        "normal_y": jnp.array(pad_src_ny),
        "dx": jnp.array(raw_dx),            # Clean 0D scalar array
        "num": jnp.array(n, dtype=jnp.int32) # Metadata tracking key for hardware masks
    }

def instantiate(scene):
    d_scene = {
        "wavelength": jnp.array(scene["wavelength"]),
    }

    def instantiate_object(obj):
        return _instantiate_object(obj, scene["samples_per_wavelength"], scene["wavelength"])
    
    d_scene["objs"] = [instantiate_object(obj) for obj in scene["objs"]]
    return d_scene


def _rayleigh_sommerfeld_batched_kernel(
    src_x, src_y, src_normal_x, src_normal_y, src_dx, src_field, src_num,
    dst_x, dst_y, dst_num,
    wavelength, batch_size: int
):
    # Retrieve compile-time constant footprints
    src_capacity = src_x.shape[0]
    dst_capacity = dst_x.shape[0]
    num_batches = dst_capacity // batch_size

    # Uniform grid-stride reshaping
    batched_dst_x = dst_x.reshape(num_batches, batch_size)
    batched_dst_y = dst_y.reshape(num_batches, batch_size)

    # Compute loop-invariant hardware predicates outside the lax.scan timeline
    dst_mask_2d = (jnp.arange(dst_capacity) < dst_num).reshape(num_batches, batch_size)
    src_mask = jnp.arange(src_capacity) < src_num

    scan_inputs = (batched_dst_x, batched_dst_y, dst_mask_2d)

    def batch_kernel(carry, inputs_slice):
        cur_dst_x, cur_dst_y, cur_dst_mask = inputs_slice

        # Compute Cartesian Product grids for this specific slice
        ax = src_x[jnp.newaxis, :]
        bx = cur_dst_x[:, jnp.newaxis]
        ay = src_y[jnp.newaxis, :]
        by = cur_dst_y[:, jnp.newaxis]

        vel_x = bx - ax
        vel_y = by - ay

        # Guard against zero-distance singularities
        l = jnp.sqrt(vel_x**2 + vel_y**2 + 1e-12)
        dir_x = vel_x / l
        dir_y = vel_y / l

        # Compute obliquity modifications
        cos_theta_line = jnp.abs(dir_x * src_normal_x[jnp.newaxis, :] + dir_y * src_normal_y[jnp.newaxis, :])
        src_is_point = src_num == 1
        cos_theta = jnp.where(src_is_point, jnp.ones_like(cos_theta_line), cos_theta_line)

        # Combine active source and destination thread indicators
        grid_mask = cur_dst_mask[:, jnp.newaxis] & src_mask[jnp.newaxis, :]

        k = 2.0 * jnp.pi / wavelength
        p = (jnp.exp(1j * k * l) / jnp.sqrt(l)) * cos_theta * src_field[jnp.newaxis, :]

        # Zero out padding contributions to keep Riemann summation unpolluted
        p = jnp.where(grid_mask, p, 0.0j)

        # Core reduction phase (src_dx is a pure 0D array scalar)
        batch_result = jnp.sum(p, axis=1) * src_dx

        # Apply Huygens-Fresnel scaling 
        norm_factor = 1.0 / jnp.sqrt(1j * wavelength)
        batch_result = batch_result * norm_factor

        # Clean out invalid trailing elements inside this batch allocation
        batch_output = jnp.where(cur_dst_mask, batch_result, 0.0j)

        return None, batch_output

    _, batched_wavefields = lax.scan(batch_kernel, None, scan_inputs)
    return batched_wavefields.reshape(-1)


def propagate(d_scene, ia, ib, d_src_field):
    """
    Main call block matching your original execution API.
    Safely routes fixed-capacity fields across the masked execution grid.
    """
    # Defensive programming: Ensure incoming field matches our structural source footprints
    # If the user passes a truncated unpadded array, pad it to _SRC_CAPACITY instantly.
    if d_src_field.shape[0] != _SRC_CAPACITY:
        d_src_field = jnp.pad(d_src_field, (0, _SRC_CAPACITY - d_src_field.shape[0]))

    fn = jax.jit(_rayleigh_sommerfeld_batched_kernel, static_argnames=["batch_size"])

    return fn(
        d_scene["objs"][ia]["src_pos_x"],
        d_scene["objs"][ia]["src_pos_y"],
        d_scene["objs"][ia]["normal_x"],
        d_scene["objs"][ia]["normal_y"],
        d_scene["objs"][ia]["dx"],
        d_src_field,
        d_scene["objs"][ia]["num"],
        d_scene["objs"][ib]["pos_x"],
        d_scene["objs"][ib]["pos_y"],
        d_scene["objs"][ib]["num"],
        d_scene["wavelength"],
        _BATCH_SIZE
    )
