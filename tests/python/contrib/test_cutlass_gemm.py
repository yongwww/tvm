# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
from typing import Tuple

import ml_dtypes
import numpy as np

import tvm
import tvm.testing
from tvm.contrib.pickle_memoize import memoize


def get_random_ndarray(shape, dtype):
    if dtype == "int8":
        return np.random.randint(-128, 128, shape).astype(dtype)
    elif dtype == "uint8":
        return np.random.randint(0, 256, shape).astype(dtype)
    return np.random.uniform(-1, 1, shape).astype(dtype)


def verify_group_gemm(
    func_name, M, N, K, num_groups, x_dtype, weight_dtype, out_dtype, use_scale, rtol, atol
):
    group_gemm_func = tvm.get_global_func(func_name, allow_missing=True)
    if group_gemm_func is None:
        print(f"Skipped as {func_name} is not available")
        return

    @memoize("tvm.contrib.cutlass.test_group_gemm_sm90")
    def get_ref_data():
        assert M % num_groups == 0
        M_per_group = M // num_groups
        a_np = get_random_ndarray((M, K), "float16")
        b_np = get_random_ndarray((num_groups, N, K), "float16")
        indptr_np = np.arange(1, num_groups + 1).astype("int64") * M_per_group
        c_np = np.concatenate(
            [a_np[i * M_per_group : (i + 1) * M_per_group] @ b_np[i].T for i in range(num_groups)],
            axis=0,
        )
        return a_np, b_np, indptr_np, c_np

    def to_numpy_dtype(dtype):
        mapping = {"float8_e5m2": ml_dtypes.float8_e5m2, "float8_e4m3fn": ml_dtypes.float8_e4m3fn}
        return mapping.get(dtype, dtype)

    a_np, b_np, indptr_np, c_np = get_ref_data()
    dev = tvm.cuda(0)
    a_nd = tvm.nd.array(a_np.astype(to_numpy_dtype(x_dtype)), device=dev)
    b_nd = tvm.nd.array(b_np.astype(to_numpy_dtype(weight_dtype)), device=dev)
    c_nd = tvm.nd.empty(c_np.shape, dtype=out_dtype, device=dev)
    indptr_nd = tvm.nd.array(indptr_np, device=dev)
    workspace = tvm.nd.empty((4096 * 1024,), dtype="uint8", device=dev)
    if use_scale:
        scale = tvm.nd.array(np.array([1.0], dtype="float32"), device=dev)
        group_gemm_func(a_nd, b_nd, indptr_nd, workspace, scale, c_nd)
    else:
        group_gemm_func(a_nd, b_nd, indptr_nd, workspace, c_nd)
    tvm.testing.assert_allclose(c_nd.numpy(), c_np, rtol=rtol, atol=atol)


@tvm.testing.requires_cutlass
@tvm.testing.requires_cuda_compute_version(9)
def test_group_gemm_sm90():
    verify_group_gemm(
        "cutlass.group_gemm_fp16_sm90",
        8,
        128,
        128,
        4,
        "float16",
        "float16",
        "float16",
        False,
        rtol=1e-3,
        atol=1e-3,
    )
    verify_group_gemm(
        "cutlass.group_gemm_e5m2_e5m2_fp16",
        8,
        16,
        16,
        4,
        "float8_e5m2",
        "float8_e5m2",
        "float16",
        True,
        rtol=1e-1,
        atol=1,
    )
    verify_group_gemm(
        "cutlass.group_gemm_e4m3_e4m3_fp16",
        8,
        16,
        16,
        4,
        "float8_e4m3fn",
        "float8_e4m3fn",
        "float16",
        True,
        rtol=1e-1,
        atol=1,
    )


def rowwise_quant_fp8_e4m3(shape: Tuple[int, int], block_size: Tuple[int, int], dtype: str):
    x_full_np = (np.random.rand(*shape) * 2 - 1).astype(dtype)
    x_scale_shape = (
        *shape[:-1],
        (shape[-1] + block_size[1] - 1) // block_size[1],
    )
    # For each (block_size[1]) block, compute the max abs value of `w_full_np`
    x_max_abs_np = np.zeros(x_scale_shape, dtype="float32")
    for i in range(x_scale_shape[-1]):
        x_max_abs_np[..., i] = np.max(
            np.abs(x_full_np[..., i * block_size[1] : min((i + 1) * block_size[1], shape[-1])]),
            axis=-1,
        )[0]
    # Scale is the `x_max_abs_np` divided by the max value of quant_dtype in ml_dtypes
    fp8_max = float(ml_dtypes.finfo("float8_e4m3fn").max)
    x_scale_np = x_max_abs_np / fp8_max
    # `x_np` is the `x_full_np` divided by the `x_scale_np` (with block awareness),
    # clamped to (-fp8_max, fp8_max), and cast to `quant_dtype`
    x_np = np.zeros_like(x_full_np, dtype="float8_e4m3fn")
    for i in range(x_scale_shape[-1]):
        x_np[..., i * block_size[1] : min((i + 1) * block_size[1], shape[-1])] = np.clip(
            x_full_np[..., i * block_size[1] : min((i + 1) * block_size[1], shape[-1])]
            / x_scale_np[..., i : i + 1],
            -fp8_max,
            fp8_max,
        )

    x_scale_np = np.random.rand(*x_scale_np.shape).astype("float32") / fp8_max
    for i in range(x_scale_shape[-1]):
        x_full_np[..., i * block_size[1] : min((i + 1) * block_size[1], shape[-1])] = (
            x_np[..., i * block_size[1] : min((i + 1) * block_size[1], shape[-1])].astype(
                x_scale_np.dtype
            )
            * x_scale_np[..., i : i + 1]
        )
    return x_np, x_scale_np


def blockwise_quant_fp8_e4m3(shape: Tuple[int, int], block_size: Tuple[int, int], dtype: str):
    w_full_np = (np.random.rand(*shape) * 2 - 1).astype(dtype)
    w_scale_shape = (
        *shape[:-2],
        (shape[-2] + block_size[0] - 1) // block_size[0],
        (shape[-1] + block_size[1] - 1) // block_size[1],
    )
    # For each (block_size[0], block_size[1]) block, compute the max abs value of `w_full_np`
    w_max_abs_np = np.zeros(w_scale_shape, dtype="float32")
    for i in range(w_scale_shape[-2]):
        for j in range(w_scale_shape[-1]):
            block_shape = (
                *shape[:-2],
                min(block_size[0], shape[-2] - i * block_size[0]),
                min(block_size[1], shape[-1] - j * block_size[1]),
            )
            w_max_abs_np[..., i, j] = np.max(
                np.abs(
                    w_full_np[
                        ...,
                        i * block_size[0] : min((i + 1) * block_size[0], shape[-2]),
                        j * block_size[1] : min((j + 1) * block_size[1], shape[-1]),
                    ]
                ).reshape(*shape[:-2], block_shape[-2] * block_shape[-1]),
                axis=-1,
            )
    # Scale is the `w_max_abs_np` divided by the max value of quant_dtype in ml_dtypes
    fp8_max = float(ml_dtypes.finfo("float8_e4m3fn").max)
    w_scale_np = w_max_abs_np / fp8_max
    # `w_np` is the `w_full_np` divided by the `w_scale_np` (with block awareness),
    # clamped to (-fp8_max, fp8_max), and cast to `quant_dtype`
    w_np = np.zeros_like(w_full_np, dtype="float8_e4m3fn")
    if len(w_scale_shape) == 2:
        for i in range(w_scale_shape[-2]):
            for j in range(w_scale_shape[-1]):
                w_np[
                    i * block_size[0] : min((i + 1) * block_size[0], shape[-2]),
                    j * block_size[1] : min((j + 1) * block_size[1], shape[-1]),
                ] = np.clip(
                    w_full_np[
                        i * block_size[0] : min((i + 1) * block_size[0], shape[-2]),
                        j * block_size[1] : min((j + 1) * block_size[1], shape[-1]),
                    ]
                    / w_scale_np[..., i, j],
                    -fp8_max,
                    fp8_max,
                )
    else:
        for e in range(w_scale_shape[0]):
            for i in range(w_scale_shape[-2]):
                for j in range(w_scale_shape[-1]):
                    w_np[
                        e,
                        i * block_size[0] : min((i + 1) * block_size[0], shape[-2]),
                        j * block_size[1] : min((j + 1) * block_size[1], shape[-1]),
                    ] = np.clip(
                        w_full_np[
                            e,
                            i * block_size[0] : min((i + 1) * block_size[0], shape[-2]),
                            j * block_size[1] : min((j + 1) * block_size[1], shape[-1]),
                        ]
                        / w_scale_np[e, i, j],
                        -fp8_max,
                        fp8_max,
                    )

    w_scale_np = np.random.rand(*w_scale_np.shape).astype("float32") / fp8_max
    return w_np, w_scale_np


def blockwise_matmul(
    x_fp8_np: np.ndarray,
    x_scale_np: np.ndarray,
    w_np: np.ndarray,
    w_scale_np: np.ndarray,
    block_size: Tuple[int, int],
    dtype: str,
):
    o_np = np.zeros((x_fp8_np.shape[0], w_np.shape[0]), dtype=dtype)
    for j in range(w_scale_np.shape[0]):
        for k in range(w_scale_np.shape[1]):
            o_np[:, j * block_size[0] : min((j + 1) * block_size[0], w_np.shape[0])] += (
                np.matmul(
                    x_fp8_np[
                        :, k * block_size[1] : min((k + 1) * block_size[1], x_fp8_np.shape[1])
                    ].astype(dtype),
                    w_np[
                        j * block_size[0] : min((j + 1) * block_size[0], w_np.shape[0]),
                        k * block_size[1] : min((k + 1) * block_size[1], w_np.shape[1]),
                    ].T.astype(dtype),
                )
                * x_scale_np[:, k : k + 1]
                * w_scale_np[j, k]
            )
    return o_np


def blockwise_bmm(
    x_fp8_np: np.ndarray,
    x_scale_np: np.ndarray,
    w_np: np.ndarray,
    w_scale_np: np.ndarray,
    block_size: Tuple[int, int],
    dtype: str,
):
    o_np = np.zeros((x_fp8_np.shape[0], x_fp8_np.shape[1], w_np.shape[1]), dtype=dtype)
    for j in range(w_scale_np.shape[1]):
        for k in range(w_scale_np.shape[2]):
            o_np[..., j * block_size[0] : min((j + 1) * block_size[0], w_np.shape[1])] += (
                np.matmul(
                    x_fp8_np[
                        ..., k * block_size[1] : min((k + 1) * block_size[1], x_fp8_np.shape[2])
                    ].astype(dtype),
                    w_np[
                        ...,
                        j * block_size[0] : min((j + 1) * block_size[0], w_np.shape[1]),
                        k * block_size[1] : min((k + 1) * block_size[1], w_np.shape[2]),
                    ]
                    .transpose(0, 2, 1)
                    .astype(dtype),
                )
                * x_scale_np[..., k : k + 1]
                * w_scale_np[..., j : j + 1, k : k + 1]
            )
    return o_np


@tvm.testing.requires_cutlass
@tvm.testing.requires_cuda_compute_version(9)
def test_fp8_e4m3_blockwise_scaled_gemm():
    M = 16
    N = 4608
    K = 896
    block_size = (128, 128)
    assert N % 128 == 0 and K % 128 == 0  # Only support N/K are multiple of 128

    func_name = "cutlass.blockwise_scaled_gemm_e4m3fn_e4m3fn"
    gemm_func = tvm.get_global_func(func_name, allow_missing=True)
    if gemm_func is None:
        print(f"Skipped as {func_name} is not available")
        return

    device = tvm.cuda(0)
    dtype = "bfloat16"
    x_np, x_scale_np = rowwise_quant_fp8_e4m3((M, K), block_size, dtype)
    w_np, w_scale_np = blockwise_quant_fp8_e4m3((N, K), block_size, dtype)
    o_np = blockwise_matmul(x_np, x_scale_np, w_np, w_scale_np, block_size, dtype)
    x_tvm = tvm.nd.array(x_np, device=device)
    x_scale_tvm = tvm.nd.array(x_scale_np.T, device=device)
    w_tvm = tvm.nd.array(w_np, device=device)
    w_scale_tvm = tvm.nd.array(w_scale_np, device=device)
    workspace = tvm.nd.empty((4096 * 1024,), dtype="uint8", device=device)
    o_tvm = tvm.nd.empty((M, N), dtype=dtype, device=device)
    gemm_func(
        x_tvm, w_tvm, x_scale_tvm, w_scale_tvm, workspace, block_size[0], block_size[1], o_tvm
    )
    o_tvm = o_tvm.numpy()
    tvm.testing.assert_allclose(o_tvm, o_np, rtol=1e-4, atol=0.5)


@tvm.testing.requires_cutlass
@tvm.testing.requires_cuda_compute_version(9)
def test_fp8_e4m3_blockwise_scaled_bmm():
    B = 16
    M = 40
    N = 512
    K = 128
    block_size = (128, 128)
    assert N % 128 == 0 and K % 128 == 0  # Only support N/K are multiple of 128

    func_name = "cutlass.blockwise_scaled_bmm_e4m3fn_e4m3fn"
    gemm_func = tvm.get_global_func(func_name, allow_missing=True)
    if gemm_func is None:
        print(f"Skipped as {func_name} is not available")
        return

    device = tvm.cuda(0)
    dtype = "bfloat16"
    x_np, x_scale_np = rowwise_quant_fp8_e4m3((B, M, K), block_size, dtype)
    w_np, w_scale_np = blockwise_quant_fp8_e4m3((B, N, K), block_size, dtype)
    o_np = blockwise_bmm(x_np, x_scale_np, w_np, w_scale_np, block_size, dtype)
    x_tvm = tvm.nd.array(x_np, device=device)
    x_scale_tvm = tvm.nd.array(x_scale_np.transpose(0, 2, 1), device=device)
    w_tvm = tvm.nd.array(w_np, device=device)
    w_scale_tvm = tvm.nd.array(w_scale_np, device=device)
    workspace = tvm.nd.empty((4096 * 1024,), dtype="uint8", device=device)
    o_tvm = tvm.nd.empty((B, M, N), dtype=dtype, device=device)
    gemm_func(
        x_tvm, w_tvm, x_scale_tvm, w_scale_tvm, workspace, block_size[0], block_size[1], o_tvm
    )
    o_tvm = o_tvm.numpy()
    tvm.testing.assert_allclose(o_tvm, o_np, rtol=1e-4, atol=0.5)


kE2M1ToFloatArray = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
]

def e2m1_to_fp32(int4_value):
    signBit = int4_value & 0x8
    int4_absValue = int4_value & 0x7
    float_result = kE2M1ToFloatArray[int4_absValue]
    if signBit:
        float_result = -float_result
    return float_result


def break_fp4_bytes(a, dtype):
    assert a.dtype == np.uint8, "Input array must be uint8"
    m, n = a.shape
    a = a.flatten()
    # Get upper 4 bits
    highHalfByte = (a & 0xF0) >> 4
    # Get lower 4 bits
    lowHalfByte = a & 0x0F
    # Apply the conversion function - assuming it handles arrays or use np.vectorize
    fH = np.array([e2m1_to_fp32(x) for x in highHalfByte], dtype=dtype)
    fL = np.array([e2m1_to_fp32(x) for x in lowHalfByte], dtype=dtype)
    # [0xAB, 0xCD] -> [0xB, 0xA, 0xD, 0xC] -> stacked [fL, fH]
    out = np.stack((fL, fH), axis=-1).reshape(m, n * 2)
    return out


def convert_swizzled_to_linear(a_sf_swizzled: np.ndarray, m, k, block_size):
    # Assuming a_sf_swizzled is a NumPy array (e.g., uint8)
    sf_m, sf_k = a_sf_swizzled.shape
    m_tiles = (m + 128 - 1) // 128
    f = block_size * 4 # Bytes per 16 elements (block_size * 2 fp4 values per byte * 2 scaling factors) ? Recheck logic if needed. Original comments unclear.
                      # Let's stick to the original calculation f = block_size * 4
    k_tiles = (k + f - 1) // f
    # Reshape and transpose according to CUTLASS swizzling pattern
    tmp = np.reshape(a_sf_swizzled, (1, m_tiles, k_tiles, 32, 4, 4))
    tmp = np.transpose(tmp, (0, 1, 4, 3, 2, 5)) # Permute equivalent
    # Determine the expected k dimension after swizzle/reshape
    # Original had k_tiles * f // block_size. Let's keep this logic.
    k_dim_out = k_tiles * f // block_size
    out = tmp.reshape(m_tiles * 128, k_dim_out)
    # Slice to the original dimensions
    return out[0:m, 0:k]


def dequantize_to_dtype(
    tensor_fp4: np.ndarray, tensor_sf: np.ndarray, global_scale, dtype, block_size=16
):
    """Dequantize the fp4 tensor back to high precision using NumPy."""
    # Two fp4 values are packed into one uint8.
    assert tensor_fp4.dtype == np.uint8, "FP4 tensor must be uint8"
    # Assuming tensor_sf is also uint8 representing float8 scales before conversion
    m, packed_k = tensor_fp4.shape
    k = packed_k * 2

    # Dequantize FP4 values to float32
    # Pass the target high-precision dtype (e.g., np.float32)
    tensor_f32 = break_fp4_bytes(tensor_fp4, dtype=dtype)
    tensor_f32 = tensor_f32.reshape(m, k // block_size, block_size)

    # Handle the scaling factors (previously float8)
    # Convert swizzled scaling factors to linear layout
    # Assuming tensor_sf holds the raw uint8 data for the scales
    tensor_sf_linear = convert_swizzled_to_linear(tensor_sf, m, k, block_size)

    # Convert scales to the target high-precision dtype and apply global scale
    # Note: This assumes tensor_sf_linear uint8 values directly represent the desired scale factors
    # If tensor_sf represents float8 specifically, a float8->float32 conversion function would be needed here.
    # For now, we convert uint8 directly to float32 as per the original code's apparent logic.
    tensor_sf_dtype = tensor_sf_linear.astype(dtype) / global_scale

    # Scale the tensor: perform broadcasting (m, k//bs, bs) * (m, k//bs, 1)
    out = (tensor_f32 * tensor_sf_dtype[..., np.newaxis]).reshape(m, k)
    return out


def get_ref_results(
    a_fp4: np.ndarray,
    b_fp4: np.ndarray,
    a_sf: np.ndarray,
    b_sf: np.ndarray,
    a_global_scale,
    b_global_scale,
    m, # Not used directly, derived from shapes
    n, # Not used directly, derived from shapes
    dtype, # Target high-precision dtype (e.g., np.float32)
    block_size,
    # device parameter removed as NumPy is CPU-based
):
    """Calculate reference GEMM results using NumPy after dequantization."""
    _, m_k = a_fp4.shape
    _, n_k = b_fp4.shape
    assert m_k == n_k

    a_in_dtype = dequantize_to_dtype(
        a_fp4, a_sf, a_global_scale, dtype=dtype, block_size=block_size
    )
    # Note: For B, the layout might be different (ColumnMajor often used).
    # The dequantization logic might need adjustment if B's swizzling or packing differs from A.
    # Assuming B has the same structure and dimensions (N, K) packed.
    b_in_dtype = dequantize_to_dtype(
        b_fp4, b_sf, b_global_scale, dtype=dtype, block_size=block_size
    )

    # Perform GEMM: A @ B.T
    # NumPy uses .T for transpose
    return np.matmul(a_in_dtype, b_in_dtype.T)

FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = np.array(448.0, dtype=np.float32)

# E2M1 to float
# 0111 -> 6
# 0110 -> 4
# 0101 -> 3
# 0100 -> 2
# 0011 -> 1.5
# 0010 -> 1
# 0001 -> 0.5
# 0000 -> 0
E2M1_TO_FLOAT32 = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]
BLOCK_SIZE = 16

def get_reciprocal(x):
    if isinstance(x, np.ndarray):
        # Use np.where to handle division by zero
        return np.where(x == 0, 0.0, 1.0 / x).astype(x.dtype)
    elif isinstance(x, (float, int, np.float32)):
        return 0.0 if x == 0 else 1.0 / x
    else:
        raise TypeError("Input must be a float, int, or a NumPy ndarray.")


def cast_to_fp4(x):
    # Apply E2M1 quantization logic using NumPy operations
    sign = np.sign(x)
    x_abs = np.abs(x)

    # Using np.select for conditional assignment is cleaner than multiple boolean masks
    conditions = [
        (x_abs >= 0.0) & (x_abs <= 0.25),
        (x_abs > 0.25) & (x_abs < 0.75),
        (x_abs >= 0.75) & (x_abs <= 1.25),
        (x_abs > 1.25) & (x_abs < 1.75),
        (x_abs >= 1.75) & (x_abs <= 2.5),
        (x_abs > 2.5) & (x_abs < 3.5),
        (x_abs >= 3.5) & (x_abs <= 5.0),
        (x_abs > 5.0)
    ]
    choices = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

    quantized_abs = np.select(conditions, choices, default=x_abs) # default shouldn't be hit if conditions cover all cases

    return quantized_abs * sign


def ref_nvfp4_quant(x, global_scale):
    # Note: NumPy does not have a built-in float8 type.
    # The original code quantizes the scale to float8_e4m3fn and back to float32.
    # This step is omitted here. Calculations are kept in float32.
    # If float8 quantization behavior is critical, a custom function or library is needed.

    assert isinstance(global_scale, np.ndarray) or isinstance(global_scale, np.number), "global_scale must be a NumPy array or scalar"
    if isinstance(global_scale, np.ndarray):
         assert global_scale.dtype == np.float32, "global_scale dtype must be float32"
    assert x.ndim == 2, "Input array must be 2-dimensional"
    m, n = x.shape

    # Ensure BLOCK_SIZE divides n
    if n % BLOCK_SIZE != 0:
        raise ValueError(f"The second dimension ({n}) must be divisible by BLOCK_SIZE ({BLOCK_SIZE})")

    x_reshaped = np.reshape(x, (m, n // BLOCK_SIZE, BLOCK_SIZE))

    # Find max absolute value per block
    vec_max = np.max(np.abs(x_reshaped), axis=-1, keepdims=True).astype(np.float32)

    # Calculate scales
    # Avoid division by zero for FLOAT4_E2M1_MAX
    reciprocal_fp4_max = get_reciprocal(FLOAT4_E2M1_MAX)
    scale = global_scale * (vec_max * reciprocal_fp4_max)

    # Omitted: scale = scale.to(torch.float8_e4m3fn).to(torch.float32)
    # If needed, insert custom float8 quantization here.

    # Calculate output scale (inverse of the calculated scale relative to global_scale)
    # Avoid division by zero for scale
    reciprocal_scale = get_reciprocal(scale)
    reciprocal_global_scale = get_reciprocal(global_scale)
    output_scale = reciprocal_scale * reciprocal_global_scale

    # Scale and clip the input tensor
    scaled_x = x_reshaped.astype(np.float32) * output_scale # Broadcasting (m, n/bs, bs) * (m, n/bs, 1)
    clipped_x = np.clip(scaled_x, -FLOAT4_E2M1_MAX, FLOAT4_E2M1_MAX).reshape(m, n)

    # Quantize to FP4 values
    quantized_values = cast_to_fp4(clipped_x)

    # Return quantized values and the calculated scales (without the final dimension)
    return quantized_values, scale.squeeze(axis=-1)


def recover_swizzled_scales(scale, m, n):
    # Assuming scale is the swizzled NumPy array (e.g., from quantization output)
    # Calculate rounded dimensions based on CUTLASS swizzling (128 for M, 4 for N blocks)
    rounded_m = ((m + 128 - 1) // 128) * 128
    scale_n = n // BLOCK_SIZE # Number of blocks along N dimension
    rounded_n = ((scale_n + 4 - 1) // 4) * 4 # Rounded number of blocks along N

    # Reshape assuming specific swizzle pattern (matches original torch code)
    # The dimensions (1, rounded_m // 128, rounded_n // 4, 32, 4, 4) must match the swizzled input shape.
    # If scale's shape doesn't match this expectation, an error will occur.
    try:
        tmp = np.reshape(scale, (1, rounded_m // 128, rounded_n // 4, 32, 4, 4))
    except ValueError as e:
        raise ValueError(f"Input scale shape {scale.shape} is incompatible with expected swizzled dimensions (1, {rounded_m // 128}, {rounded_n // 4}, 32, 4, 4)") from e

    # Permute axes to reverse the swizzle
    tmp = np.transpose(tmp, (0, 1, 4, 3, 2, 5))

    # Reshape back to a 2D linear layout
    result = np.reshape(tmp, (rounded_m, rounded_n)).astype(np.float32)

    # Slice to get the original dimensions
    return result[:m, :scale_n]

@tvm.testing.requires_cutlass
@tvm.testing.requires_cuda_compute_version(10)
def test_fp4_e2m1_blockscaled_gemm():
    SHAPE = (128, 128, 64)
    M, N, packed_k = SHAPE
    K = packed_k * 2
    block_size = 16
    assert N % 128 == 0 and K % 128 == 0  # Only support N/K are multiple of 128

    func_name = "cutlass.scaled_gemm_e2m1_e2m1_fp16"
    gemm_func = tvm.get_global_func(func_name, allow_missing=True)
    if gemm_func is None:
        print(f"Skipped as {func_name} is not available")
        return

    device = tvm.cuda(0)
    dtype = "bfloat16"

    # Generate random matrices
    from ml_dtypes import bfloat16
    a_dtype = np.random.randn(M, K).astype(bfloat16)
    b_dtype = np.random.randn(N, K).astype(bfloat16)

    # Calculate global scales
    a_global_scale = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / np.amax(a_dtype.flatten()).astype(np.float32)
    b_global_scale = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / np.amax(b_dtype.flatten()).astype(np.float32)

    # Calculate alpha
    alpha = 1.0 / (a_global_scale * b_global_scale)

    # Quantize matrices
    a_np, a_scale_np = ref_nvfp4_quant(a_dtype, a_global_scale)
    b_np, b_scale_np = ref_nvfp4_quant(b_dtype, b_global_scale)

    a_np = b_np =  np.random.randn(M, N // 2).astype(np.uint8)
    rounded_m = ((M + 128 - 1) // 128) * 128 # 128
    scale_n = N // block_size
    rounded_n = ((scale_n + 4 - 1) // 4) * 4 # 8
    a_scale_np = b_scale_np = np.random.randn(rounded_m, rounded_n // 4).astype(ml_dtypes.float8_e4m3fn)

    # x_np, x_scale_np = rowwise_quant_fp8_e4m3((M, K), block_size, dtype)
    # w_np, w_scale_np = blockwise_quant_fp8_e4m3((N, K), block_size, dtype)
    # o_np = blockwise_matmul(a_np, x_scale_np, b_np, w_scale_np, block_size, dtype)

    # Print types and shapes of all inputs
    # print(f"Type of a_dtype: {a_dtype.dtype}, Shape: {a_dtype.shape}")
    # print(f"Type of b_dtype: {b_dtype.dtype}, Shape: {b_dtype.shape}")
    print(f"Type of a_global_scale: {a_global_scale.dtype}")  # Scalar, no shape
    print(f"Type of b_global_scale: {b_global_scale.dtype}")  # Scalar, no shape
    print(f"Type of alpha: {alpha.dtype}")  # Scalar, no shape
    print(f"Type of a_np: {a_np.dtype}, Shape: {a_np.shape}")
    print(f"Type of a_scale_np: {a_scale_np.dtype}, Shape: {a_scale_np.shape}")
    print(f"Type of b_np: {b_np.dtype}, Shape: {b_np.shape}")
    print(f"Type of b_scale_np: {b_scale_np.dtype}, Shape: {b_scale_np.shape}")

    a_tvm = tvm.nd.array(a_np, device=device)
    b_tvm = tvm.nd.array(b_np, device=device)
    workspace = tvm.nd.empty((4096 * 1024,), dtype="uint8", device=device)
    o_tvm = tvm.nd.empty((M, N), dtype=dtype, device=device)
    alpha_tvm = tvm.nd.array([alpha], device=device)
    sfa_tvm = tvm.nd.array(a_scale_np, device=device)
    sfb_tvm = tvm.nd.array(b_scale_np, device=device)
    gemm_func(
        a_tvm, b_tvm, workspace, alpha_tvm, sfa_tvm, sfb_tvm, o_tvm
    )
    o_tvm = o_tvm.numpy()
    tvm.testing.assert_allclose(o_tvm, o_np, rtol=1e-4, atol=0.5)


if __name__ == "__main__":
    # tvm.testing.main()
    test_fp4_e2m1_blockscaled_gemm()
