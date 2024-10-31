import time
from typing import Dict, List, Tuple, Union
import numpy as np
import torch
import flashinfer
import argparse
import tvm
from tvm import tir, relax
from tvm.relax.frontend.nn import core, modules, spec
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T
from tvm.relax.frontend.nn.llm import kv_cache
from tvm import dlight as dl
from tvm.runtime import ShapeTuple
import mlc_llm
from mlc_llm.compiler_pass import pipeline
from mlc_llm.nn.kv_cache import PagedKVCache, RopeMode

"""
paged_kv_cache: R.Object
t0: R.Tensor((seq_len, 48/tp, 128), dtype="float16")
a0 = R.call_dps_packed("vm.builtin.attention_kv_cache_attention_with_fused_qkv", (paged_kv_cache, R.prim_value(0), R.prim_value(T.float32(1.0)), t0), out_sinfo=R.Tensor((seq_len, 32/tp, 128), dtype="float16"))
a1 = R.call_dps_packed("vm.builtin.attention_kv_cache_attention_with_fused_qkv", (paged_kv_cache, R.prim_value(1), R.prim_value(T.float32(1.0)), t0), out_sinfo=R.Tensor((seq_len, 32/tp, 128), dtype="float16"))
a2 = R.call_dps_packed("vm.builtin.attention_kv_cache_attention_with_fused_qkv", (paged_kv_cache, R.prim_value(2), R.prim_value(T.float32(1.0)), t0), out_sinfo=R.Tensor((seq_len, 32/tp, 128), dtype="float16"))
...
a31 = R.call_dps_packed("vm.builtin.attention_kv_cache_attention_with_fused_qkv", (paged_kv_cache, R.prim_value(31), R.prim_value(T.float32(1.0)), t0), out_sinfo=R.Tensor((seq_len, 32/tp, 128), dtype="float16"))
"""

##########
reserved_nseq = 32
maximum_total_seq_length = 2048
prefill_chunk_size = 512
page_size = 16
num_layers =32
num_qo_heads = 32
num_kv_heads = 8
head_dim = 128
rope_scale = 1.0 # todo (yongwww): check
rope_theta = 500000.0
dtype = "float16"
device = tvm.cuda()

fclear = None
fcreate = None
fadd_sequence = None
fremove_sequence = None
ffork_sequence = None
fpopn = None
fbegin_forward = None
fend_forward = None
fattention = None
fattention_with_fuse_qkv = None
fdebug_get_kv = None

fattention_prefill = None
fattention_decode = None
fattention_prefill_ragged = None
fattention_prefill_begin_forward = None
fattention_prefill_end_forward = None
fattention_decode_begin_forward = None
fattention_decode_end_forward = None
fattention_prefill_ragged_begin_forward = None
fattention_prefill_ragged_end_forward = None
fattention_merge_state = None

ftranspose_append = None
fsplit_rotary = None
fcopy_single_page = None
fcopy_cache = None


@T.prim_func
def kv_cache_transpose_append(
    var_pages: T.handle,
    var_k_data: T.handle,
    var_v_data: T.handle,
    var_position_map: T.handle,
):
    ntoken = T.SizeVar("ntoken", "int64")
    page_size = T.SizeVar("page_size", "int64")
    num_pages = T.int64()
    position_map_elem_offset = T.int32()
    pages = T.match_buffer(var_pages, (num_pages, 2, num_kv_heads, page_size, head_dim), dtype)
    k_data = T.match_buffer(var_k_data, (ntoken, num_kv_heads, head_dim), dtype)
    v_data = T.match_buffer(var_v_data, (ntoken, num_kv_heads, head_dim), dtype)
    position_map = T.match_buffer(
        var_position_map, (ntoken,), "int32", elem_offset=position_map_elem_offset
    )

    for global_pos, h, f in T.grid(ntoken, num_kv_heads, head_dim):
        with T.block("k_transpose_append"):
            vgpos, vh, vf = T.axis.remap("SSS", [global_pos, h, f])
            T.reads(position_map[vgpos], k_data[vgpos, vh, vf])
            T.writes(
                pages[position_map[vgpos] // page_size, 0, vh, position_map[vgpos] % page_size, vf]
            )
            position: T.int64 = T.Cast("int64", position_map[vgpos])
            pages[
                T.floordiv(position, page_size), 0, vh, T.floormod(position, page_size), vf
            ] = k_data[vgpos, vh, vf]
        with T.block("v_transpose_append"):
            vgpos, vh, vf = T.axis.remap("SSS", [global_pos, h, f])
            T.reads(position_map[vgpos], k_data[vgpos, vh, vf])
            T.writes(
                pages[position_map[vgpos] // page_size, 1, vh, position_map[vgpos] % page_size, vf]
            )
            position: T.int64 = T.Cast("int64", position_map[vgpos])
            pages[
                T.floordiv(position, page_size), 1, vh, T.floormod(position, page_size), vf
            ] = v_data[vgpos, vh, vf]


def llama_rope_with_position_map(  # pylint: disable=too-many-arguments
    theta: float,
    scale: float,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    dtype: float = "float16",
    rotary_dim: int = None,
):
    fused_heads = num_q_heads + num_kv_heads * 2
    if rotary_dim is None:
        rotary_dim = head_dim
    scale = tir.const(scale, dtype)

    def _rope_freq(s: tir.Var, d: tir.Var, d_range: int, theta: float, dtype: str):
        freq = s / tir.power(theta, d * 2 % d_range / tir.const(d_range, "float32"))
        cos_freq = tir.cos(freq).astype(dtype)
        sin_freq = tir.sin(freq).astype(dtype)
        return cos_freq, sin_freq

    def _rope(  # pylint: disable=too-many-arguments
        x: T.Buffer,
        s: tir.Var,
        h: tir.Var,
        d: tir.Var,
        pos: tir.Var,
    ):
        cos_freq, sin_freq = _rope_freq(pos * scale, d, rotary_dim, theta, dtype)
        cos = cos_freq * x[s, h, d]
        sin = sin_freq * tir.if_then_else(
            d < rotary_dim // 2,
            -x[s, h, d + rotary_dim // 2],
            x[s, h, d - rotary_dim // 2],
        )
        return cos + sin

    @T.prim_func(private=True)
    def fused_rope(  # pylint: disable=too-many-locals
        var_qkv: T.handle,
        var_position_map: T.handle,
        var_q: T.handle,
        var_k: T.handle,
        var_v: T.handle,
        apply_rope: T.int32,
    ):
        T.func_attr(
            {
                "op_pattern": 8,  # 2 means injective, 8 means opaque
                "tir.noalias": T.bool(True),
            }
        )
        seq_len = T.int64()
        position_map_elem_offset = T.int64()
        qkv = T.match_buffer(var_qkv, (seq_len, fused_heads, head_dim), dtype)
        q = T.match_buffer(var_q, (seq_len, num_q_heads, head_dim), dtype)
        k = T.match_buffer(var_k, (seq_len, num_kv_heads, head_dim), dtype)
        v = T.match_buffer(var_v, (seq_len, num_kv_heads, head_dim), dtype)
        position_map = T.match_buffer(
            var_position_map, (seq_len,), "int32", elem_offset=position_map_elem_offset
        )
        for iters in T.grid(seq_len, fused_heads, head_dim):
            with T.block("llama_fused_rope"):
                s, h, d = T.axis.remap("SSS", iters)
                if h < num_q_heads:
                    q[s, h, d] = T.if_then_else(
                        apply_rope > 0 and d < rotary_dim,
                        _rope(qkv, s, h, d, position_map[s]),
                        qkv[s, h, d],
                    )
                elif h < num_q_heads + num_kv_heads:
                    k[s, h - num_q_heads, d] = T.if_then_else(
                        apply_rope > 0 and d < rotary_dim,
                        _rope(qkv, s, h, d, position_map[s]),
                        qkv[s, h, d],
                    )
                else:
                    v[s, h - (num_q_heads + num_kv_heads), d] = qkv[s, h, d]

    return fused_rope


@T.prim_func
def copy_cache(
    var_pages: T.handle,
    var_position_map: T.handle,
    var_k_data: T.handle,
    var_v_data: T.handle,
    layer_id: T.int64,
):
    num_kv_heads = T.int64()
    head_dim = T.int64()
    seqlen = T.SizeVar("seqlen", "int64")
    page_size = T.int64()
    num_pages = T.int64()
    position_map_elem_offset = T.int64()
    pages = T.match_buffer(var_pages, (num_pages, 2, num_kv_heads, page_size, head_dim), "float16")
    position_map = T.match_buffer(
        var_position_map, (seqlen,), "int32", elem_offset=position_map_elem_offset
    )
    k_data = T.match_buffer(var_k_data, (num_layers, seqlen, num_kv_heads, head_dim), "float16")
    v_data = T.match_buffer(var_v_data, (num_layers, seqlen, num_kv_heads, head_dim), "float16")

    for p, h, d in T.grid(seqlen, num_kv_heads, head_dim):
        with T.block("copy0"):
            vp, vh, vd = T.axis.remap("SSS", [p, h, d])
            T.reads(
                position_map[vp],
                pages[position_map[vp] // page_size, 0:2, vh, position_map[vp] % page_size, vd],
            )
            T.writes(k_data[layer_id, vp, vh, vd], v_data[layer_id, vp, vh, vd])
            position: T.int64 = T.Cast("int64", position_map[vp])
            k_data[layer_id, vp, vh, vd] = pages[
                T.floordiv(position, page_size), 0, vh, T.floormod(position, page_size), vd
            ]
            v_data[layer_id, vp, vh, vd] = pages[
                T.floordiv(position, page_size), 1, vh, T.floormod(position, page_size), vd
            ]


def _copy_single_page(num_heads, page_size, head_dim, dtype, target):
    tx = 256 if str(target.kind) == "webgpu" else 1024

    @T.prim_func
    def copy_single_page(
        pages: T.handle,
        src_page_id: T.int64,
        tgt_page_id: T.int64,
        copy_length: T.int64,
    ):
        T.func_attr({"tir.is_scheduled": 1})
        num_pages = T.int32()
        P = T.match_buffer(pages, (num_pages, 2, num_heads, page_size, head_dim), dtype)

        for b in T.thread_binding(
            (copy_length * num_heads * head_dim + tx - 1) // tx, thread="blockIdx.x"
        ):
            for t in T.thread_binding(tx, thread="threadIdx.x"):
                with T.block("copy"):
                    T.where(b * tx + t < copy_length * num_heads * head_dim)
                    vh = T.axis.spatial(
                        num_heads,
                        T.Cast("int32", (b * tx + t) // (copy_length * head_dim)),
                    )
                    vp = T.axis.spatial(
                        copy_length,
                        (b * tx + t) % (copy_length * head_dim) // head_dim,
                    )
                    vd = T.axis.spatial(
                        head_dim,
                        T.Cast(
                            "int32",
                            (b * tx + t) % head_dim,
                        ),
                    )
                    P[tgt_page_id, 0, vh, vp, vd] = P[src_page_id, 0, vh, vp, vd]
                    P[tgt_page_id, 1, vh, vp, vd] = P[src_page_id, 1, vh, vp, vd]

    return copy_single_page


def set_global_func():
    global fclear, fcreate, fadd_sequence, fremove_sequence, ffork_sequence, fpopn
    global fbegin_forward, fend_forward, fattention, fattention_with_fuse_qkv, fdebug_get_kv
    global fattention_prefill, fattention_prefill_begin_forward, fattention_prefill_end_forward
    global fattention_decode, fattention_decode_begin_forward, fattention_decode_end_forward
    global fattention_prefill_ragged
    global fattention_prefill_ragged_begin_forward
    global fattention_prefill_ragged_end_forward
    global fattention_merge_state, fsplit_rotary, fcopy_single_page
    global ftranspose_append, fcopy_cache

    fclear = tvm.get_global_func("vm.builtin.kv_state_clear")
    fcreate = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_create")
    fadd_sequence = tvm.get_global_func("vm.builtin.kv_state_add_sequence")
    fremove_sequence = tvm.get_global_func("vm.builtin.kv_state_remove_sequence")
    ffork_sequence = tvm.get_global_func("vm.builtin.kv_state_fork_sequence")
    fpopn = tvm.get_global_func("vm.builtin.kv_state_popn")
    fbegin_forward = tvm.get_global_func("vm.builtin.kv_state_begin_forward")
    fend_forward = tvm.get_global_func("vm.builtin.kv_state_end_forward")
    fattention_with_fuse_qkv = tvm.get_global_func(
        "vm.builtin.attention_kv_cache_attention_with_fused_qkv"
    )
    fdebug_get_kv = tvm.get_global_func("vm.builtin.attention_kv_cache_debug_get_kv")

    fattention_prefill = tvm.get_global_func(
        "flashinfer.attention_kernel_prefill_with_paged_kv_cache"
    )
    fattention_decode = tvm.get_global_func(
        "flashinfer.attention_kernel_decode_with_paged_kv_cache"
    )
    fattention_prefill_ragged = tvm.get_global_func(
        "flashinfer.attention_kernel_prefill_with_ragged_kv_cache"
    )
    fattention_prefill_begin_forward = tvm.get_global_func(
        "flashinfer.attention_kernel_prefill_with_paged_kv_cache_begin_forward"
    )
    fattention_prefill_end_forward = tvm.get_global_func(
        "flashinfer.attention_kernel_prefill_with_paged_kv_cache_end_forward"
    )
    fattention_decode_begin_forward = tvm.get_global_func(
        "flashinfer.attention_kernel_decode_with_paged_kv_cache_begin_forward"
    )
    fattention_decode_end_forward = tvm.get_global_func(
        "flashinfer.attention_kernel_decode_with_paged_kv_cache_end_forward"
    )
    fattention_prefill_ragged_begin_forward = tvm.get_global_func(
        "flashinfer.attention_kernel_prefill_with_ragged_kv_cache_begin_forward"
    )
    fattention_prefill_ragged_end_forward = tvm.get_global_func(
        "flashinfer.attention_kernel_prefill_with_ragged_kv_cache_end_forward"
    )
    fattention_merge_state = tvm.get_global_func("flashinfer.merge_state_in_place")

    target = tvm.target.Target.from_device(device)
    builts = []
    for tir_func in [
        kv_cache_transpose_append,
        llama_rope_with_position_map(
            rope_theta, rope_scale, head_dim, num_qo_heads, num_kv_heads, dtype
        ),
        _copy_single_page(num_kv_heads, page_size, head_dim, dtype, target),
        copy_cache,
    ]:
        mod = tvm.IRModule({"main": tir_func})
        with target:
            mod = dl.ApplyDefaultSchedule(dl.gpu.Fallback())(mod)
        f = tvm.build(mod["main"], target=target)
        builts.append(f.entry_func)

    ftranspose_append, fsplit_rotary, fcopy_single_page, fcopy_cache = builts


def create_kv_cache(rope_mode):
    support_sliding_window = 0
    cache = fcreate(
        tvm.runtime.ShapeTuple(
            [
                reserved_nseq,
                maximum_total_seq_length,
                prefill_chunk_size,
                page_size,
                support_sliding_window,
            ]
        ),
        tvm.runtime.ShapeTuple([0, num_layers]),
        num_qo_heads,
        num_kv_heads,
        head_dim,
        rope_mode,
        rope_scale,
        rope_theta,
        tvm.nd.empty((), dtype, device=device),
        ftranspose_append,
        fattention_prefill,
        fattention_decode,
        fattention_prefill,
        fattention_decode,
        fattention_prefill_ragged,
        fattention_prefill_ragged_begin_forward,
        fattention_prefill_ragged_end_forward,
        fattention_prefill_begin_forward,
        fattention_prefill_end_forward,
        fattention_decode_begin_forward,
        fattention_decode_end_forward,
        fattention_merge_state,
        fsplit_rotary,
        fcopy_single_page,
        fcopy_cache,
        None,
        None,
        None,
    )
    return cache
##########


def test_nn_module_paged_kv_cache():
    # fmt: off
    @I.ir_module
    class Module:
        @R.function
        def create_paged_kv_cache(
            max_batch_size: R.Shape(["max_batch_size_1"]),  # type: ignore
            max_total_seq_len: R.Shape(["max_total_seq_len_1"]),  # type: ignore
            prefill_chunk_size: R.Shape(["prefill_chunk_size_1"]),  # type: ignore
            page_size: R.Shape(["page_size_1"]),  # type: ignore
            support_sliding_window: R.Shape(["support_sliding_window_1"]),  # type: ignore
        ) -> R.Object:
            max_batch_size_1 = T.int64()
            max_total_seq_len_1 = T.int64()
            prefill_chunk_size_1 = T.int64()
            page_size_1 = T.int64()
            support_sliding_window_1 = T.int64()
            R.func_attr({"num_input": 5})

            #$ paged_kv_cache: R.Object = R.call_pure_packed("mlc.create_paged_kv_cache_generic", R.shape([max_batch_size, max_total_seq_len, prefill_chunk_size, page_size, support_sliding_window]         ), R.shape([0, 32]), R.prim_value(32), R.prim_value(32), R.prim_value(8), R.prim_value(128), R.prim_value(1), R.prim_value(1), R.prim_value(T.float32(500000.0)), R.str("{}"), R.prim_value(0), R.prim_value(128), R.dtype("float16"), sinfo_args=(R.Object,))

            with R.dataflow():
                paged_kv_cache: R.Object = R.call_pure_packed("mlc.create_paged_kv_cache_generic", R.shape([max_batch_size_1, max_total_seq_len_1, prefill_chunk_size_1, page_size_1, support_sliding_window_1]), R.shape([0, 32]), R.prim_value(32), R.prim_value(32), R.prim_value(32), R.prim_value(128), R.prim_value(1), R.prim_value(1), R.prim_value(10000),  R.str("{}"), R.prim_value(0), R.prim_value(128), R.dtype("float16"), sinfo_args=(R.Object,))
                gv1: R.Object = paged_kv_cache
                R.output(gv1)
            return gv1

        @R.function
        def forward(
            cache: R.Object, qkv: R.Tensor((1, 100, 96, 128), dtype="float16")  # type: ignore
        ) -> R.Tensor((1, 100, 32, 128), dtype="float16"):  # type: ignore
            R.func_attr({"num_input": 2})
            with R.dataflow():
                reshape: R.Tensor((100, 96, 128), dtype="float16") = R.reshape(  # type: ignore
                    qkv, R.shape([100, 96, 128])
                )
                lv = R.call_dps_packed(
                    "vm.builtin.attention_kv_cache_attention_with_fused_qkv",
                    (cache, R.prim_value(0), R.prim_value(T.float32(1)), reshape),
                    out_sinfo=R.Tensor((100, 32, 128), dtype="float16"),
                )
                reshape1: R.Tensor((1, 100, 32, 128), dtype="float16") = R.reshape(  # type: ignore
                    lv, R.shape([1, 100, 32, 128])
                )
                gv: R.Tensor((1, 100, 32, 128), dtype="float16") = reshape1  # type: ignore
                R.output(gv)
            return gv
    # fmt: on

    class PagedKVCacheTest(modules.Module):
        def forward(
            self,
            cache: PagedKVCache,
            qkv: core.Tensor,
        ) -> core.Tensor:
            return cache.attention_with_fused_qkv(0, qkv, num_qo_heads=32)

        def create_paged_kv_cache(
            self,
            max_batch_size: tir.Var,
            max_total_seq_len: tir.Var,
            prefill_chunk_size: tir.Var,
            page_size: tir.Var,
            support_sliding_window: tir.Var,
        ) -> PagedKVCache:
            return PagedKVCache.create_generic(
                max_batch_size=max_batch_size,
                max_total_seq_len=max_total_seq_len,
                prefill_chunk_size=prefill_chunk_size,
                page_size=page_size,
                support_sliding_window=support_sliding_window,
                num_hidden_layers=32,
                num_attention_heads=32,
                num_key_value_heads=32,
                head_dim=128,
                rope_mode=RopeMode.NORMAL,
                # rope_mode=RopeMode.INLINE,
                rope_scale=1,
                rope_theta=10000,
                rotary_dim=128,
                dtype="float16",
            )

    export_results = PagedKVCacheTest().export_tvm(
        spec={
            "forward": {
                "cache": spec.Object(object_type=PagedKVCache),
                "qkv": spec.Tensor((1, 100, 96, 128), "float16"),
            },
            "create_paged_kv_cache": {
                "max_batch_size": int,
                "max_total_seq_len": int,
                "prefill_chunk_size": int,
                "page_size": int,
                "support_sliding_window": int,
            },
        },
    )
    tvm_mod = export_results[0]
    # tvm.ir.assert_structural_equal(tvm_mod, Module, True)

    mod = tvm_mod
    mod.show()
    tgt = tvm.target.Target("nvidia/nvidia-h100")
    dev = tvm.gpu()
    """
    mod = relax.transform.LegalizeOps()(mod)
    with tgt:
        mod = tvm.tir.transform.DefaultGPUSchedule()(mod)
    ex = relax.build(mod, tgt)
    print(ex.as_text())
    vm = relax.VirtualMachine(ex, dev)
    """

    pipeline=relax.get_pipeline("mlc_llm", target=tgt, flashinfer=True, cublas_gemm=True, faster_transformer=True)
    with tgt:
        mod = pipeline(mod)
    ex = relax.build(mod, tgt)

    print(ex.as_text())
    vm = relax.VirtualMachine(ex, dev)

    # inference
    max_batch_size = tvm.nd.array(1, dev)

    # cache = vm['create_flashinfer_paged_kv_cache'](tvm.runtime.ShapeTuple([10]), tvm.runtime.ShapeTuple([2048]), tvm.runtime.ShapeTuple([512]), tvm.runtime.ShapeTuple([16]), tvm.runtime.ShapeTuple([1]))
    # cache = vm['create_flashinfer_paged_kv_cache'](max_batch_size_=7, max_total_seq_len_=100, prefill_chunk_size_=10, page_size_=10, support_sliding_window_=1)

    cache = vm['create_tir_paged_kv_cache'](tvm.runtime.ShapeTuple([10]), tvm.runtime.ShapeTuple([2048]), tvm.runtime.ShapeTuple([512]), tvm.runtime.ShapeTuple([16]), tvm.runtime.ShapeTuple([1]))
    res = vm['forward'](cache, tvm.nd.array(np.random.rand(1, 100, 96, 128).astype(np.float16), dev))


def test_prefill_attention():
    @I.ir_module
    class Module:
        @R.function
        def create_paged_kv_cache(max_batch_size_1: R.Shape(["max_batch_size"]), max_total_seq_len_1: R.Shape(["max_total_seq_len"]), prefill_chunk_size_1: R.Shape(["prefill_chunk_size"]), page_size_1: R.Shape(["page_size"]), support_sliding_window_1: R.Shape(["support_sliding_window"])) -> R.Object:
            max_batch_size = T.int64()
            max_total_seq_len = T.int64()
            prefill_chunk_size = T.int64()
            page_size = T.int64()
            support_sliding_window = T.int64()
            R.func_attr({"num_input": 5})
            with R.dataflow():
                paged_kv_cache: R.Object = R.call_pure_packed("mlc.create_paged_kv_cache_generic", R.shape([max_batch_size, max_total_seq_len, prefill_chunk_size, page_size, support_sliding_window]), R.shape([0, 32]), R.prim_value(32), R.prim_value(32), R.prim_value(8), R.prim_value(128), R.prim_value(1), R.prim_value(1), R.prim_value(T.float32(500000.0)), R.str("{}"), R.prim_value(0), R.prim_value(128), R.dtype("float16"), sinfo_args=(R.Object,))
                gv13: R.Object = paged_kv_cache
                R.output(gv13)
            return gv13
        @R.function
        def forward(cache: R.Object, qkv: R.Tensor(("seq_len", 48, 128), dtype="float16")) -> R.Tensor(("seq_len", 32, 128), dtype="float16"):
            seq_len = T.int64()
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv0 = R.call_dps_packed("vm.builtin.attention_kv_cache_attention_with_fused_qkv", (cache, R.prim_value(0), R.prim_value(T.float32(1.0)), qkv), out_sinfo=R.Tensor((seq_len, 32, 128), dtype="float16"))
                lv1 = R.call_dps_packed("vm.builtin.attention_kv_cache_attention_with_fused_qkv", (cache, R.prim_value(1), R.prim_value(T.float32(1.0)), qkv), out_sinfo=R.Tensor((seq_len, 32, 128), dtype="float16"))
                lv2 = R.call_dps_packed("vm.builtin.attention_kv_cache_attention_with_fused_qkv", (cache, R.prim_value(2), R.prim_value(T.float32(1.0)), qkv), out_sinfo=R.Tensor((seq_len, 32, 128), dtype="float16"))
                lv3 = R.call_dps_packed("vm.builtin.attention_kv_cache_attention_with_fused_qkv", (cache, R.prim_value(3), R.prim_value(T.float32(1.0)), qkv), out_sinfo=R.Tensor((seq_len, 32, 128), dtype="float16"))
                lv4 = R.call_dps_packed("vm.builtin.attention_kv_cache_attention_with_fused_qkv", (cache, R.prim_value(4), R.prim_value(T.float32(1.0)), qkv), out_sinfo=R.Tensor((seq_len, 32, 128), dtype="float16"))
                lv5 = R.call_dps_packed("vm.builtin.attention_kv_cache_attention_with_fused_qkv", (cache, R.prim_value(5), R.prim_value(T.float32(1.0)), qkv), out_sinfo=R.Tensor((seq_len, 32, 128), dtype="float16"))
                lv6 = R.call_dps_packed("vm.builtin.attention_kv_cache_attention_with_fused_qkv", (cache, R.prim_value(6), R.prim_value(T.float32(1.0)), qkv), out_sinfo=R.Tensor((seq_len, 32, 128), dtype="float16"))
                lv7 = R.call_dps_packed("vm.builtin.attention_kv_cache_attention_with_fused_qkv", (cache, R.prim_value(7), R.prim_value(T.float32(1.0)), qkv), out_sinfo=R.Tensor((seq_len, 32, 128), dtype="float16"))
                lv8 = R.call_dps_packed("vm.builtin.attention_kv_cache_attention_with_fused_qkv", (cache, R.prim_value(8), R.prim_value(T.float32(1.0)), qkv), out_sinfo=R.Tensor((seq_len, 32, 128), dtype="float16"))
                lv9 = R.call_dps_packed("vm.builtin.attention_kv_cache_attention_with_fused_qkv", (cache, R.prim_value(9), R.prim_value(T.float32(1.0)), qkv), out_sinfo=R.Tensor((seq_len, 32, 128), dtype="float16"))
                lv10 = R.call_dps_packed("vm.builtin.attention_kv_cache_attention_with_fused_qkv", (cache, R.prim_value(10), R.prim_value(T.float32(1.0)), qkv), out_sinfo=R.Tensor((seq_len, 32, 128), dtype="float16"))
                lv11 = R.call_dps_packed("vm.builtin.attention_kv_cache_attention_with_fused_qkv", (cache, R.prim_value(11), R.prim_value(T.float32(1.0)), qkv), out_sinfo=R.Tensor((seq_len, 32, 128), dtype="float16"))
                lv12 = R.call_dps_packed("vm.builtin.attention_kv_cache_attention_with_fused_qkv", (cache, R.prim_value(12), R.prim_value(T.float32(1.0)), qkv), out_sinfo=R.Tensor((seq_len, 32, 128), dtype="float16"))
                lv13 = R.call_dps_packed("vm.builtin.attention_kv_cache_attention_with_fused_qkv", (cache, R.prim_value(13), R.prim_value(T.float32(1.0)), qkv), out_sinfo=R.Tensor((seq_len, 32, 128), dtype="float16"))
                lv14 = R.call_dps_packed("vm.builtin.attention_kv_cache_attention_with_fused_qkv", (cache, R.prim_value(14), R.prim_value(T.float32(1.0)), qkv), out_sinfo=R.Tensor((seq_len, 32, 128), dtype="float16"))
                lv15 = R.call_dps_packed("vm.builtin.attention_kv_cache_attention_with_fused_qkv", (cache, R.prim_value(15), R.prim_value(T.float32(1.0)), qkv), out_sinfo=R.Tensor((seq_len, 32, 128), dtype="float16"))
                lv16 = R.call_dps_packed("vm.builtin.attention_kv_cache_attention_with_fused_qkv", (cache, R.prim_value(16), R.prim_value(T.float32(1.0)), qkv), out_sinfo=R.Tensor((seq_len, 32, 128), dtype="float16"))
                lv17 = R.call_dps_packed("vm.builtin.attention_kv_cache_attention_with_fused_qkv", (cache, R.prim_value(17), R.prim_value(T.float32(1.0)), qkv), out_sinfo=R.Tensor((seq_len, 32, 128), dtype="float16"))
                lv18 = R.call_dps_packed("vm.builtin.attention_kv_cache_attention_with_fused_qkv", (cache, R.prim_value(18), R.prim_value(T.float32(1.0)), qkv), out_sinfo=R.Tensor((seq_len, 32, 128), dtype="float16"))
                lv19 = R.call_dps_packed("vm.builtin.attention_kv_cache_attention_with_fused_qkv", (cache, R.prim_value(19), R.prim_value(T.float32(1.0)), qkv), out_sinfo=R.Tensor((seq_len, 32, 128), dtype="float16"))
                lv20 = R.call_dps_packed("vm.builtin.attention_kv_cache_attention_with_fused_qkv", (cache, R.prim_value(20), R.prim_value(T.float32(1.0)), qkv), out_sinfo=R.Tensor((seq_len, 32, 128), dtype="float16"))
                lv21 = R.call_dps_packed("vm.builtin.attention_kv_cache_attention_with_fused_qkv", (cache, R.prim_value(21), R.prim_value(T.float32(1.0)), qkv), out_sinfo=R.Tensor((seq_len, 32, 128), dtype="float16"))
                lv22 = R.call_dps_packed("vm.builtin.attention_kv_cache_attention_with_fused_qkv", (cache, R.prim_value(22), R.prim_value(T.float32(1.0)), qkv), out_sinfo=R.Tensor((seq_len, 32, 128), dtype="float16"))
                lv23 = R.call_dps_packed("vm.builtin.attention_kv_cache_attention_with_fused_qkv", (cache, R.prim_value(23), R.prim_value(T.float32(1.0)), qkv), out_sinfo=R.Tensor((seq_len, 32, 128), dtype="float16"))
                lv24 = R.call_dps_packed("vm.builtin.attention_kv_cache_attention_with_fused_qkv", (cache, R.prim_value(24), R.prim_value(T.float32(1.0)), qkv), out_sinfo=R.Tensor((seq_len, 32, 128), dtype="float16"))
                lv25 = R.call_dps_packed("vm.builtin.attention_kv_cache_attention_with_fused_qkv", (cache, R.prim_value(25), R.prim_value(T.float32(1.0)), qkv), out_sinfo=R.Tensor((seq_len, 32, 128), dtype="float16"))
                lv26 = R.call_dps_packed("vm.builtin.attention_kv_cache_attention_with_fused_qkv", (cache, R.prim_value(26), R.prim_value(T.float32(1.0)), qkv), out_sinfo=R.Tensor((seq_len, 32, 128), dtype="float16"))
                lv27 = R.call_dps_packed("vm.builtin.attention_kv_cache_attention_with_fused_qkv", (cache, R.prim_value(27), R.prim_value(T.float32(1.0)), qkv), out_sinfo=R.Tensor((seq_len, 32, 128), dtype="float16"))
                lv28 = R.call_dps_packed("vm.builtin.attention_kv_cache_attention_with_fused_qkv", (cache, R.prim_value(28), R.prim_value(T.float32(1.0)), qkv), out_sinfo=R.Tensor((seq_len, 32, 128), dtype="float16"))
                lv29 = R.call_dps_packed("vm.builtin.attention_kv_cache_attention_with_fused_qkv", (cache, R.prim_value(29), R.prim_value(T.float32(1.0)), qkv), out_sinfo=R.Tensor((seq_len, 32, 128), dtype="float16"))
                lv30 = R.call_dps_packed("vm.builtin.attention_kv_cache_attention_with_fused_qkv", (cache, R.prim_value(30), R.prim_value(T.float32(1.0)), qkv), out_sinfo=R.Tensor((seq_len, 32, 128), dtype="float16"))
                lv31 = R.call_dps_packed("vm.builtin.attention_kv_cache_attention_with_fused_qkv", (cache, R.prim_value(31), R.prim_value(T.float32(1.0)), qkv), out_sinfo=R.Tensor((seq_len, 32, 128), dtype="float16"))

                # lv129 = R.call_dps_packed("vm.builtin.attention_kv_cache_attention_with_fused_qkv", (paged_kv_cache, R.prim_value(0), R.prim_value(T.float32(1.0)), reshape513), out_sinfo=R.Tensor((seq_len, 32, 128), dtype="float16"))
                gv: R.Tensor((seq_len, 32, 128), dtype="float16") = lv0
                R.output(gv)
            return gv

    mod = Module
    tgt = tvm.target.Target("nvidia/nvidia-h100")
    pipeline=relax.get_pipeline("mlc_llm", target=tgt, flashinfer=True, cublas_gemm=True, faster_transformer=True)
    dev = tvm.gpu()
    with tgt:
        mod = pipeline(mod)
    mod['forward'].show()
    # print(mod.script())
    ex = relax.build(mod, tgt)

    # print(f"ex.entry_func: {ex.entry_func}")

    # print(ex.as_text())
    vm = relax.VirtualMachine(ex, dev)

    # inference
    max_batch_size = tvm.nd.array(1, dev)

    # fsplit_rotary = vm['fused_rope']()
    set_global_func()
    cache = create_kv_cache(RopeMode.NORMAL)
    # operation_seq = [[(0, 6)], [(1, 8)], [(2, 11)], [(3, 16)], [(4, 19), (5, 20)]]
    operation_seq = [[(100, 100)],]
    seq_ids = []
    append_lengths = []
    for batch in operation_seq:
        for seq_id, append_length in batch:
            seq_ids.append(seq_id)
            append_lengths.append(append_length)
            fadd_sequence(cache, seq_id)
            # vm['add_sequence'](cache, seq_id)
    fbegin_forward(cache, ShapeTuple(seq_ids), ShapeTuple(append_lengths))
    # print(f"yongwww cache = {cache}, type = {type(cache)}") yongwww cache = relax.vm.PagedAttentionKVCache(0x7db2cc8), type = <class 'tvm.runtime.object.Object'>
    # return

    #cache = vm['create_flashinfer_paged_kv_cache'](tvm.runtime.ShapeTuple([10]), tvm.runtime.ShapeTuple([2048]), tvm.runtime.ShapeTuple([512]), tvm.runtime.ShapeTuple([32]), tvm.runtime.ShapeTuple([1]))

    # print(f"cache = {cache}")
    # cache = vm['create_flashinfer_paged_kv_cache'](max_batch_size_=7, max_total_seq_len_=100, prefill_chunk_size_=10, page_size_=10, support_sliding_window_=1)
    # cache = vm['create_tir_paged_kv_cache'](tvm.runtime.ShapeTuple([10]), tvm.runtime.ShapeTuple([2048]), tvm.runtime.ShapeTuple([512]), tvm.runtime.ShapeTuple([16]), tvm.runtime.ShapeTuple([1]))
    iterations = 10000
    warmup = 100
    nd_qkv = tvm.nd.array(np.random.rand(3000, 48, 128).astype(np.float16), dev)

    # collect nsys
    return_nsys = False
    if return_nsys:
        import cuda
        import cuda.cudart
        # start profiling
        cuda.cudart.cudaProfilerStart()
        res = vm['forward'](cache, nd_qkv)
        cuda.cudart.cudaProfilerStop()
        print("profiled")
        return

    # with relax vm
    for _ in range(warmup):
       res = vm['forward'](cache, nd_qkv)

    start_time = time.time_ns()
    for _ in range(iterations):
        res = vm['forward'](cache, nd_qkv)
    dur = time.time_ns() - start_time
    print(f"avg time of vm forward: {dur/iterations/1e3} us")
    return

    # single vm buildtin call
    outputs = tvm.nd.empty((3000, 32, 128), dtype, device=dev)
    for _ in range(warmup):
       fattention_with_fuse_qkv(cache, 0, 1.0, nd_qkv, outputs)

    start_time = time.time_ns()
    for _ in range(iterations):
        fattention_with_fuse_qkv(cache, 0, 1.0, nd_qkv, outputs)
    dur = time.time_ns() - start_time
    print(f"avg time of fattention_with_fuse_qkv: {dur/iterations/1e3} us")

    # res = vm['forward'](cache, tvm.nd.array(np.random.rand(1, 100, 48, 128).astype(np.float16), dev))


def bench_flashinfer_batch_prefill_with_paged_kv_cache(
    batch_size=7,
    num_layers=32,
    num_qo_heads=32,
    num_kv_heads=8,
    head_dim=128,
    page_size=16,
    nnz_qo=100,
    kv_len=54,
    kv_layout="NHD",
    tensor_parallel_shards=1,
):
    num_qo_heads //= tensor_parallel_shards
    num_kv_heads //= tensor_parallel_shards
    # Allocate 128MB workspace buffer
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout
    )
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    qo_indptr = torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32)
    kv_indptr = torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32) * num_pages_per_seq
    kv_indices = torch.arange(0, total_num_pages, device="cuda:0", dtype=torch.int32)
    kv_last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32, device="cuda:0"
    )

    q_at_layer = torch.randn(num_layers, nnz_qo, num_qo_heads, head_dim, dtype=torch.float16, device="cuda:0")
    kv_cache_at_layer = torch.randn(
        num_layers, total_num_pages, 2, page_size, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0"
    )

    # Create auxiliary data structures for batch prefill attention
    prefill_wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=True,
    )

    outputs = []
    iterations = 100000
    start_time = time.monotonic_ns()

    for i in range(num_layers):
        q = q_at_layer[i]
        kv_data = kv_cache_at_layer[i]
        for _ in range(iterations):
            o = prefill_wrapper.run(q, kv_data)
        outputs.append(o)

    dur = (time.monotonic_ns() - start_time) / iterations / num_layers / 1000
    print(f"Average flashinfer prefill attention with paged kv_cache took: {dur:.2f} μs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark FlashInfer Batch Prefill with Paged KV Cache")
    parser.add_argument("--batch_size", type=int, default=7, help="Batch size for the benchmark")
    parser.add_argument("--tensor_parallel_shards", type=int, default=1, help="Number of tensor parallel shards")
    parser.add_argument("--seq_len", type=int, default=300, help="The sequence length")

    args = parser.parse_args()
    test_prefill_attention()

    # bench_flashinfer_batch_prefill_with_paged_kv_cache(
    #    batch_size=args.batch_size,
    #    tensor_parallel_shards=args.tensor_parallel_shards,
    #    nnz_qo=args.seq_len,
    #    kv_len=args.seq_len)
