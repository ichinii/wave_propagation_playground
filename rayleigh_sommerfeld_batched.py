import sys
import jax
import jax.numpy as jnp
import jax.lax as lax

"""
    dst_capacity must be divisible by batch_size
"""
def rayleigh_sommerfeld_batched_kernel(
    src_x, src_y, src_normal_x, src_normal_y, src_dx, src_data, src_num,
    dst_x, dst_y, dst_num,
    wavelength, batch_size: int
):
    src_capacity = src_x.shape[0]
    dst_capacity = dst_x.shape[0]
    
    num_batches = dst_capacity // batch_size
    
    batched_dst_x = dst_x.reshape(num_batches, batch_size)
    batched_dst_y = dst_y.reshape(num_batches, batch_size)
    
    dst_mask_2d = (jnp.arange(dst_capacity) < dst_num).reshape(num_batches, batch_size)
    
    # CRITICAL COMPILER OPTIMIZATION: Hoist loop-invariant mask calculation out of the scan body
    src_mask = jnp.arange(src_capacity) < src_num
    
    scan_inputs = (batched_dst_x, batched_dst_y, dst_mask_2d)

    def batch_kernel(carry, inputs_slice):
        cur_dst_x, cur_dst_y, cur_dst_mask = inputs_slice
        
        ax = src_x[jnp.newaxis, :]
        bx = cur_dst_x[:, jnp.newaxis]
        ay = src_y[jnp.newaxis, :]
        by = cur_dst_y[:, jnp.newaxis]

        vel_x = bx - ax
        vel_y = by - ay
        
        l = jnp.sqrt(vel_x**2 + vel_y**2 + 1e-12)
        dir_x = vel_x / l
        dir_y = vel_y / l

        cos_theta_line = jnp.abs(dir_x * src_normal_x[jnp.newaxis, :] + dir_y * src_normal_y[jnp.newaxis, :])
        src_is_point = src_num == 1
        cos_theta = jnp.where(src_is_point, jnp.ones_like(cos_theta_line), cos_theta_line)

        # Clean closure lookup of the pre-computed src_mask
        grid_mask = cur_dst_mask[:, jnp.newaxis] & src_mask[jnp.newaxis, :]

        k = 2.0 * jnp.pi / wavelength
        p = (jnp.exp(1j * k * l) / jnp.sqrt(l)) * cos_theta * src_data[jnp.newaxis, :]

        p = jnp.where(grid_mask, p, 0.0j)

        batch_result = jnp.sum(p, axis=1) * src_dx
        
        norm_factor = 1.0 / jnp.sqrt(1j * wavelength)
        batch_result = batch_result * norm_factor

        batch_output = jnp.where(cur_dst_mask, batch_result, 0.0j)

        return None, batch_output

    _, batched_wavefields = lax.scan(batch_kernel, None, scan_inputs)

    return batched_wavefields.reshape(-1)

def rayleigh_sommerfeld_batched(scene, ia, ib, s):
    buckets = [1024, 2048, 4096, 8192, 16384, 32768, 65536]

    def get_nearest_bucked(num):
        for b in buckets:
            if num <= b:
                return b
        sys.exit(f"Error: num {num} exceeds maximum bucket size {buckets[-1]}")

    src_capacity = get_nearest_bucked(scene["num"][ia])
    dst_capacity = get_nearest_bucked(scene["num"][ib])

    fn = jax.jit(fun=rayleigh_sommerfeld_batched_kernel, static_argnames=["batch_size"])
    return fn(
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
        jnp.array(scene["wavelength"]),
        128
    )
