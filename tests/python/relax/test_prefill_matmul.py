from typing import List
import tvm
from tvm import relax
import numpy as np

import tvm.script
from tvm.script import ir as I, tir as T, relax as R
from tvm.relax.frontend.nn import core, modules, spec, op
import mlc_llm
from mlc_llm.compiler_pass import pipeline
import argparse
import os
import time
import pandas as pd

"""
t0: R.Tensor((1, seq_len, 4096), dtype=dtype)
t1: R.Tensor((4096, 6144/tp), dtype=dtype)
m0: R.Tensor((1, seq_len, 6144/tp), dtype=dtype) = R.matmul(t0, t1)

t2: R.Tensor((1, seq_len, 4096/tp), dtype=dtype)
t3: R.Tensor((4096/tp, 4096), dtype=dtype)
m1: R.Tensor((1, seq_len, 4096), dtype=dtype) = R.matmul(t2, t3)

t4: R.Tensor((1, seq_len, 4096), dtype=dtype)
t5: R.Tensor((4096, 28672/tp), dtype=dtype)
m2: R.Tensor((1, seq_len, 28672/tp), dtype=dtype) = R.matmul(t4, t5)

t6: R.Tensor((1, seq_len, 28672/ tp / 2), dtype=dtype)
t7: R.Tensor((28672/ tp / 2, 4096), dtype=dtype)
m3: R.Tensor((1, seq_len, 4096), dtype=dtype) = R.matmul(t6, t7)

# Report the above pattern for num_hidden_layers = 32 times
#  batch prefill
l1: R.Tensor((1, batch_size, 4096), dtype=dtype)
l2: R.Tensor((4096, vocab_size), dtype=dtype)
m4: R.Tensor((1, batch_size, vocab_size), dtype=dtype) = R.matmul(l1, l2)
"""


def handle_result(time_list: List[int]):
    df = pd.DataFrame(time_list)
    mean = df.mean()[0]
    min = df.min()[0]
    max = df.max()[0]
    std = df.std()[0]
    return round(mean / 1e6, 2), round(min / 1e6, 2), round(max / 1e6, 2), round(std / 1e6, 2)


def test_prefill_matmul(batch=1, tp=1, seq_len=300, dtype="float16"):
    num_layers = 32
    class LinearModule(modules.Module):
        def forward(
            self,
            t0: core.Tensor,
            t1: core.Tensor,
            t2: core.Tensor,
            t3: core.Tensor,
            t4: core.Tensor,
            t5: core.Tensor,
            t6: core.Tensor,
            t7: core.Tensor,
        ) -> core.Tensor:
            ret = []
            for _ in range(1): # this is for a single layer
                ret.append(op.matmul(t0, t1))
                ret.append(op.matmul(t2, t3))
                ret.append(op.matmul(t4, t5))
                ret.append(op.matmul(t6, t7))
            return ret

    export_results = LinearModule().export_tvm(
        spec={
            "forward": {
                "t0": spec.Tensor((batch, seq_len, 4096), "float16"),
                "t1": spec.Tensor((4096, tp + 1), "float16"),
                "t2": spec.Tensor((batch, seq_len, 4096 // tp), "float16"),
                "t3": spec.Tensor((4096 // tp, 4096), "float16"),
                "t4": spec.Tensor((batch, seq_len, 4096), "float16"),
                "t5": spec.Tensor((4096, 28672 // tp), "float16"),
                "t6": spec.Tensor((batch, seq_len, 28672 // tp // 2), "float16"),
                "t7": spec.Tensor((28672 // 2 // tp, 4096), "float16"),
            },
        },
    )
    mod = export_results[0]
    # mod.show()
    tgt = tvm.target.Target("nvidia/nvidia-h100")
    dev = tvm.gpu()
    pipeline=relax.get_pipeline("mlc_llm", target=tgt, flashinfer=True, cublas_gemm=True, faster_transformer=True)
    with tgt:
        mod = pipeline(mod)
    ex = relax.build(mod, tgt)
    # print(ex.as_text())
    vm = relax.VirtualMachine(ex, dev)

    # inference
    t0 = tvm.nd.array(np.random.rand(batch, seq_len, 4096).astype(dtype), dev)
    t1 = tvm.nd.array(np.random.rand(4096, tp + 1).astype(dtype), dev)
    t2 = tvm.nd.array(np.random.rand(batch, seq_len, 4096 // tp).astype(dtype), dev)
    t3 = tvm.nd.array(np.random.rand(4096 // tp, 4096).astype(dtype), dev)
    t4 = tvm.nd.array(np.random.rand(batch, seq_len, 4096).astype(dtype), dev)
    t5 = tvm.nd.array(np.random.rand(4096, 28672 // tp).astype(dtype), dev)
    t6 = tvm.nd.array(np.random.rand(batch, seq_len, 28672 // tp // 2).astype(dtype), dev)
    t7 = tvm.nd.array(np.random.rand(28672 // 2 // tp, 4096).astype(dtype), dev)
    res  = vm["forward"](t0, t1, t2, t3, t4, t5, t6, t7)
    # print(res[0].numpy())


    iterations = 300
    warmup = 20

    # with relax vm
    for _ in range(warmup):
       res = vm["forward"](t0, t1, t2, t3, t4, t5, t6, t7)

    results = []
    for _ in range(iterations):
        start_time = time.time_ns()
        res = vm["forward"](t0, t1, t2, t3, t4, t5, t6, t7)
        results.append((time.time_ns() - start_time) * num_layers)

    ret = handle_result(results)
    print(f"mean time of vm forward: {ret[0]} ms, min: {ret[1]}, max: {ret[2]}, "
          f"std: {ret[3]},  with seq_len = {seq_len}, tp = {tp}, batch = {batch}")
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test LLAMA 3.1 Prefill Matmul")
    parser.add_argument("--tensor_parallel_shards", type=int, default=1, help="Tensor Parallelism Factor")
    parser.add_argument("--seq_len", type=str, default="300", help="Comma-separated list of sequence lengths")
    parser.add_argument("--output", type=str, default="out_prefill_matmul.csv", help="Output file")
    parser.add_argument("--batch_size", type=str, default="1", help="Batch size")

    args = parser.parse_args()
    seq_lens = [int(s) for s in args.seq_len.split(',')]
    batch_sizes = [int(s) for s in args.batch_size.split(',')]

    # Check if results.csv exists
    if not os.path.exists(args.output):
        # Write header
        with open(args.output, 'w') as f:
            f.write('tensor_parallel_shards,batch_size,seq_len,mean_ms,min_ms,max_ms,std\n')

    for batch in batch_sizes:
       for seq_len in seq_lens:
            durations = test_prefill_matmul(seq_len=seq_len, tp=args.tensor_parallel_shards, batch=batch)
            # Append results to csv
            with open(args.output, 'a') as f:
                f.write(f'{args.tensor_parallel_shards},{batch},{seq_len},{durations[0]},{durations[1]},{durations[2]},{durations[3]}\n')
