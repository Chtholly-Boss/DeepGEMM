import torch
from typing import Tuple

from ..jit import build
from .gemm_swapAB import get_best_configs
from .runtime import (
    FP8GemmSwapABRuntime, GemmType,
    make_2d_tma_a_desc, make_2d_tma_b_desc,
    make_2d_tma_d_desc, make_2d_tma_scales_desc)
from .utils import ceil_div, get_col_major_tma_aligned_tensor, get_num_sms


def m_grouped_gemm_fp8_fp8_bf16_nt_contiguous_swapAB(lhs: Tuple[torch.Tensor, torch.Tensor],
                                                     rhs: Tuple[torch.Tensor, torch.Tensor],
                                                     out: torch.Tensor, m_indices: torch.Tensor) -> None:
    """
    Perform a grouped GEMM (contiguous format) with FP8 inputs and BF16 output, with 1x128 LHS scaling and 128x128 RHS scaling.

    Requirements:
        LHS, RHS, RHS scaling factors, and output tensors must be in contiguous format.
        RHS and RHS scaling factors are required to be transposed.
        The LHS scaling tensor requires a TMA-aligned transposed format, if your input does not match the requirement,
            this function will do a transposing with a set of slow PyTorch operations.
        On the M axis, inputs are grouped into several batches, of which batch sizes aligned to
            `get_m_alignment_for_contiguous_layout()` (128).

    Arguments:
        lhs: the first element is an FP8 tensor (typed `torch.float8_e4m3fn`) of shape `[m_sum, k]`,
             the second element is an FP32 1x128 scaling tensor for LHS of shape `[m_sum, ⌈k / 128⌉]`.
        rhs: the first element is an FP8 tensor (typed `torch.float8_e4m3fn`) of shape `[num_groups, n, k]`,
             the second element is an FP32 128x128 scaling tensor for RHS of shape `[num_groups, ⌈n / 128⌉, ⌈k / 128⌉]`.
        out: the BF16 output tensor of shape `[m_sum, n]`, representing the result.
        m_indices: a tensor of shape `[m_sum]` with type `torch.int`.
            `m_indices[i]` records the group which the i-th row of the LHS belongs to,
            which means that the i-th row of the LHS matrix will be multiplied with `rhs[m_indices[i]]`.
            Values of `m_indices` in every-m-alignment-block must also be the same.
    """
    lhs, lhs_scales = lhs
    rhs, rhs_scales = rhs
    m, k = lhs.shape
    num_groups, n, k_ = rhs.shape
    m_, n_ = out.shape
    m__ = m_indices.numel()

    # Type and shape checks
    assert m == m_ == m__ and k == k_ and n == n_
    assert lhs_scales.shape == (m, ceil_div(k, 128))
    assert rhs_scales.shape == (num_groups, ceil_div(n, 128), ceil_div(k, 128))
    assert lhs.dtype == torch.float8_e4m3fn and lhs_scales.dtype == torch.float32
    assert rhs.dtype == torch.float8_e4m3fn and rhs_scales.dtype == torch.float32
    assert out.dtype == torch.bfloat16
    assert m_indices.dtype == torch.int32
    assert lhs.is_contiguous() and rhs.is_contiguous()
    assert out.is_contiguous() and m_indices.is_contiguous()

    # LHS scales must be transposed for TMA load, but not for RHS scales
    lhs_scales = get_col_major_tma_aligned_tensor(lhs_scales)
    assert rhs_scales.is_contiguous()

    # Do nothing if `m` is zero
    if m == 0:
        return

    # Auto-tuning with compilation
    num_sms = get_num_sms()
    num_sms, block_m, block_n, num_stages, tma_multicast_config, smem_config = get_best_configs(
        m, n, k, 1, num_sms)
    block_k = 128
    num_tma_threads = 128
    num_math_threads_per_group = 128

    tensor_map_a = make_2d_tma_b_desc(GemmType.GroupedContiguousSwapAB, rhs, n, k, k, block_m, block_k, num_groups)
    tensor_map_b = make_2d_tma_a_desc(GemmType.GroupedContiguousSwapAB, lhs, m, k, k, block_n, block_k, num_groups)
    tensor_map_d = make_2d_tma_d_desc(GemmType.GroupedContiguousSwapAB, out, m, n, n, block_n, block_m, num_groups, 0)
    tensor_map_scales_a = make_2d_tma_scales_desc(GemmType.GroupedContiguousSwapAB, lhs_scales, m, k, block_n, block_k, num_groups)
    
    kwargs = {
        # Templated arguments
        'NUM_TMA_THREADS': num_tma_threads,
        'NUM_MATH_THREADS_PER_GROUP': num_math_threads_per_group,
        'M': n, 'N': m, 'K': k,
        'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k,
        'SWIZZLE_D_MODE': 0,
        'BLOCK_N_PADDING': smem_config[2],
        'NUM_GROUPS': num_groups,
        'NUM_STAGES': num_stages,
        'NUM_TMA_MULTICAST': tma_multicast_config[0],
        'IS_TMA_MULTICAST_ON_A': tma_multicast_config[1],
        'GEMM_TYPE': GemmType.GroupedContiguousSwapAB,
        # Runtime arguments
        'SCALES_B': rhs_scales,
        'GROUPED_LAYOUT': m_indices,
        'NUM_SMS': num_sms,
        'SMEM_SIZE': smem_config[0],
        'TENSOR_MAP_A': tensor_map_a,
        'TENSOR_MAP_B': tensor_map_b,
        'TENSOR_MAP_SCALES_A': tensor_map_scales_a,
        'TENSOR_MAP_D': tensor_map_d,
        'STREAM': torch.cuda.current_stream().cuda_stream,
        'DEVICE_INDEX': out.device.index
    }

    # Generate, build and run the kernel
    code = FP8GemmSwapABRuntime.generate(kwargs)
    runtime = build('m_grouped_gemm_fp8_fp8_bf16_nt_swapAB', code, FP8GemmSwapABRuntime, kwargs)
    runtime(**kwargs)
