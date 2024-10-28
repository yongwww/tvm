import time
import torch
import flashinfer
import argparse

"""
paged_kv_cache: R.Object
t0: R.Tensor((seq_len, 48/tp, 128), dtype="float16") 
a0 = R.call_dps_packed("vm.builtin.attention_kv_cache_attention_with_fused_qkv", (paged_kv_cache, R.prim_value(0), R.prim_value(T.float32(1.0)), t0), out_sinfo=R.Tensor((seq_len, 32/tp, 128), dtype="float16"))
a1 = R.call_dps_packed("vm.builtin.attention_kv_cache_attention_with_fused_qkv", (paged_kv_cache, R.prim_value(1), R.prim_value(T.float32(1.0)), t0), out_sinfo=R.Tensor((seq_len, 32/tp, 128), dtype="float16"))
a2 = R.call_dps_packed("vm.builtin.attention_kv_cache_attention_with_fused_qkv", (paged_kv_cache, R.prim_value(2), R.prim_value(T.float32(1.0)), t0), out_sinfo=R.Tensor((seq_len, 32/tp, 128), dtype="float16"))
...
a31 = R.call_dps_packed("vm.builtin.attention_kv_cache_attention_with_fused_qkv", (paged_kv_cache, R.prim_value(31), R.prim_value(T.float32(1.0)), t0), out_sinfo=R.Tensor((seq_len, 32/tp, 128), dtype="float16"))
"""

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
    parser.add_argument("--nnz_qo", type=int, default=300, help="Number of non-zero query/output tokens")

    args = parser.parse_args()

    bench_flashinfer_batch_prefill_with_paged_kv_cache(
        batch_size=args.batch_size,
        tensor_parallel_shards=args.tensor_parallel_shards,
        nnz_qo=args.nnz_qo
    )