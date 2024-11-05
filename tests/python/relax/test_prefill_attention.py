from typing import List
import time
import numpy as np
import argparse
import tvm
from tvm import tir, relax
from tvm.relax.frontend.nn import core, modules, spec
from tvm.runtime import ShapeTuple
import mlc_llm
from mlc_llm.compiler_pass import pipeline
from mlc_llm.nn.kv_cache import PagedKVCache, RopeMode
import os
import pandas as pd

"""
############################################################################################################
    paged_kv_cache: R.Object
    t0: R.Tensor((seq_len, 48/tp, 128), dtype="float16")
    a0 = R.call_dps_packed("vm.builtin.attention_kv_cache_attention_with_fused_qkv", (paged_kv_cache, R.prim_value(0), R.prim_value(T.float32(1.0)), t0), out_sinfo=R.Tensor((seq_len, 32/tp, 128), dtype="float16"))
    a1 = R.call_dps_packed("vm.builtin.attention_kv_cache_attention_with_fused_qkv", (paged_kv_cache, R.prim_value(1), R.prim_value(T.float32(1.0)), t0), out_sinfo=R.Tensor((seq_len, 32/tp, 128), dtype="float16"))
    a2 = R.call_dps_packed("vm.builtin.attention_kv_cache_attention_with_fused_qkv", (paged_kv_cache, R.prim_value(2), R.prim_value(T.float32(1.0)), t0), out_sinfo=R.Tensor((seq_len, 32/tp, 128), dtype="float16"))
    # repeat the pattern above for num_layers times
    a31 = R.call_dps_packed("vm.builtin.attention_kv_cache_attention_with_fused_qkv", (paged_kv_cache, R.prim_value(31), R.prim_value(T.float32(1.0)), t0), out_sinfo=R.Tensor((seq_len, 32/tp, 128), dtype="float16"))
############################################################################################################
"""


def handle_result(time_list: List[int]):
    df = pd.DataFrame(time_list)
    mean = df.mean()[0]
    min = df.min()[0]
    max = df.max()[0]
    std = df.std()[0]
    return round(mean / 1e3, 2), round(min / 1e3, 2), round(max / 1e3, 2), round(std / 1e3, 2)


def test_prefill_attention(seq_len=100, tensor_parallel_shards=1):

    num_layers = 32
    class PagedKVCacheModule(modules.Module):
        def forward(
            self,
            cache: PagedKVCache,
            qkv: core.Tensor,
        ) -> core.Tensor:
            ret = []
            for i in range(num_layers):
                ret.append(cache.attention_with_fused_qkv(i, qkv, num_qo_heads=32 // tensor_parallel_shards))
            return ret
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
                num_attention_heads=32 //tensor_parallel_shards,
                num_key_value_heads=8 // tensor_parallel_shards,
                head_dim=128,
                rope_mode=RopeMode.NORMAL,
                rope_scale=1.0,
                rope_theta=500000.0,
                rotary_dim=128,
                dtype="float16",
            )

    export_results = PagedKVCacheModule().export_tvm(
        spec={
            "forward": {
                "cache": spec.Object(object_type=PagedKVCache),
                "qkv": spec.Tensor((1, seq_len, 48 // tensor_parallel_shards, 128), "float16"),
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
    mod = export_results[0]
    tgt = tvm.target.Target("nvidia/nvidia-h100")
    dev = tvm.gpu()
    fadd_sequence = tvm.get_global_func("vm.builtin.kv_state_add_sequence")
    fbegin_forward = tvm.get_global_func("vm.builtin.kv_state_begin_forward")
    fend_forward = tvm.get_global_func("vm.builtin.kv_state_end_forward")
    pipeline=relax.get_pipeline("mlc_llm", target=tgt, flashinfer=True, cublas_gemm=True, faster_transformer=True)
    with tgt:
        mod = pipeline(mod)
    ex = relax.build(mod, tgt)

    # print(ex.as_text())
    vm = relax.VirtualMachine(ex, dev)

    # inference
    max_batch_size = 256
    max_total_seq_len = 32768 # 2048 # 128k
    prefill_chunk_size = 32768 # 16384 # 8192 # 16384
    page_size = 16
    support_sliding_window = 1

    cache = vm['create_flashinfer_paged_kv_cache'](
        tvm.runtime.ShapeTuple([max_batch_size]),
        tvm.runtime.ShapeTuple([max_total_seq_len]),
        tvm.runtime.ShapeTuple([prefill_chunk_size]),
        tvm.runtime.ShapeTuple([page_size]),
        tvm.runtime.ShapeTuple([support_sliding_window])
    )
    operation_seq = [[(0, seq_len)],]
    seq_ids = []
    append_lengths = []

    for batch in operation_seq:
        for seq_id, append_length in batch:
            seq_ids.append(seq_id)
            append_lengths.append(append_length)
            fadd_sequence(cache, seq_id)
    fbegin_forward(cache, ShapeTuple(seq_ids), ShapeTuple(append_lengths))
    iterations = 1000
    warmup = 100
    nd_qkv = tvm.nd.array(np.random.rand(1, seq_len, 48//tensor_parallel_shards, 128).astype(np.float16), dev)

    # with relax vm
    for _ in range(warmup):
       res = vm['forward'](cache, nd_qkv)

    results = []
    for _ in range(iterations):
        start_time = time.time_ns()
        res = vm['forward'](cache, nd_qkv)
        results.append(time.time_ns() - start_time)

    ret = handle_result(results)
    print(f"mean time of vm forward: {ret[0]} us, min: {ret[1]}, max: {ret[2]}, "
          f"std: {ret[3]},  with seq_len = {seq_len}, tp = {tensor_parallel_shards}")
    fend_forward(cache)
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark FlashInfer Batch Prefill with Paged KV Cache")
    parser.add_argument("--tensor_parallel_shards", type=int, default=1, help="Number of tensor parallel shards")
    parser.add_argument("--seq_len", type=str, default="300", help="Comma-separated list of sequence lengths")
    parser.add_argument("--output", type=str, default="out_prefill_attention.csv", help="Output file")

    args = parser.parse_args()
    seq_lens = [int(s) for s in args.seq_len.split(',')]

    # Check if results.csv exists
    if not os.path.exists(args.output):
        # Write header
        with open(args.output, 'w') as f:
            f.write('tensor_parallel_shards,seq_len,mean_us,min_us,max_us,std\n')

    for seq_len in seq_lens:
        durations = test_prefill_attention(seq_len=seq_len, tensor_parallel_shards=args.tensor_parallel_shards)
        # Append results to csv
        with open(args.output, 'a') as f:
            f.write(f'{args.tensor_parallel_shards},{seq_len},{durations[0]},{durations[1]},{durations[2]},{durations[3]}\n')