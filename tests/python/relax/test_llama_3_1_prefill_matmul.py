import tvm
from tvm import relax
import numpy as np

import tvm.script
from tvm.script import ir as I, tir as T, relax as R
import mlc_llm
from mlc_llm.compiler_pass import pipeline
import argparse

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


def test_llama_3_1_prefill_matmul(tp=1, seq_len=300, dtype="float16"):
    @tvm.script.ir_module
    class PrefillMatmul:
        @R.function
        def foo(x: R.Prim(value="tp"),
                t0: R.Tensor((1, "seq_len", 4096), "float16"),
                t1: R.Tensor((4096, "tp + 1"), "float16"),
                t2: R.Tensor((1, "seq_len", "4096 // tp"), "float16"),
                t3: R.Tensor(("4096 // tp", 4096), "float16"),
                t4: R.Tensor((1, "seq_len", 4096), "float16"),
                t5: R.Tensor((4096, "28672 // tp"), "float16"),
                t6: R.Tensor((1, "seq_len", "28672 // tp // 2"), "float16"),
                t7: R.Tensor(("28672 // 2 // tp", 4096), "float16"),
        ) -> R.Tensor((1, "seq_len", 4096), "float16"):
            seq_len = T.int64()
            tp = T.int64()
            with R.dataflow():
                m0 = R.matmul(t0, t1)
                m1 = R.matmul(t2, t3)
                m2 = R.matmul(t4, t5)
                m3 = R.matmul(t6, t7)
                R.output(m3)
            return m3
    PrefillMatmul.show()
    # TODO (yongwww): Consider to use SLM since we can control the dtype there
    # compile the model
    mod = PrefillMatmul
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
    # print(ex.as_text())
    vm = relax.VirtualMachine(ex, dev)

    # inference
    t0 = tvm.nd.array(np.random.rand(1, seq_len, 4096).astype(dtype), dev)
    t1 = tvm.nd.array(np.random.rand(4096, tp + 1).astype(dtype), dev)
    t2 = tvm.nd.array(np.random.rand(1, seq_len, 4096 // tp).astype(dtype), dev)
    t3 = tvm.nd.array(np.random.rand(4096 // tp, 4096).astype(dtype), dev)
    t4 = tvm.nd.array(np.random.rand(1, seq_len, 4096).astype(dtype), dev)
    t5 = tvm.nd.array(np.random.rand(4096, 28672 // tp).astype(dtype), dev)
    t6 = tvm.nd.array(np.random.rand(1, seq_len, 28672 // tp // 2).astype(dtype), dev)
    t7 = tvm.nd.array(np.random.rand(28672 // 2 // tp, 4096).astype(dtype), dev)
    res  = vm["foo"](tp, t0, t1, t2, t3, t4, t5, t6, t7)
    print(res.numpy())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test LLAMA 3.1 Prefill Matmul")
    parser.add_argument("--tensor_parallel_shards", type=int, default=1, help="Tensor Parallelism Factor")
    parser.add_argument("--seq_len", type=int, default=300, help="Sequence Length")
    args = parser.parse_args()
    test_llama_3_1_prefill_matmul(tp=args.tensor_parallel_shards, seq_len=args.seq_len)