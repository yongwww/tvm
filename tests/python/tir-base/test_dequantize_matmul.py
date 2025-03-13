import tvm
from tvm import relax
from tvm.ir.module import IRModule
from tvm.script import tir as T, relax as R, ir as I
import mlc_llm
from mlc_llm.compiler_pass import pipeline
import numpy as np
from ml_dtypes import float4_e2m1fn, float8_e4m3fn
import time

@I.ir_module
class Module:
    @T.prim_func
    def dequantize4(model_layers_0_mlp_down_proj_q_weight2: T.Buffer((T.int64(4096), T.int64(1792)), "uint32"), model_layers_0_mlp_down_proj_q_group_scale2: T.Buffer((T.int64(4096), T.int64(448)), "float8_e4m3fn"), model_layers_0_mlp_down_proj_q_global_scale2: T.Buffer((T.int64(1),), "float32"), dequantize: T.Buffer((T.int64(4096), T.int64(14336)), "float16")):
        T.func_attr({"target": T.target({"arch": "sm_101", "keys": ["cuda", "gpu"], "kind": "cuda", "libs": ["thrust"], "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.noalias": T.bool(True)})
        # with T.block("root"):
        compute = T.alloc_buffer((T.int64(4096), T.int64(14336)), "float16")
        for i0, i1 in T.grid(T.int64(4096), T.int64(14336)):
            with T.block("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(model_layers_0_mlp_down_proj_q_weight2[v_i0, v_i1 // T.int64(8)])
                T.writes(compute[v_i0, v_i1])
                compute[v_i0, v_i1] = T.Cast("float16", T.reinterpret("float4_e2m1fn", T.Cast("uint8", T.bitwise_and(T.shift_right(model_layers_0_mlp_down_proj_q_weight2[v_i0, v_i1 // T.int64(8)], T.Cast("uint32", v_i1 % T.int64(8) * T.int64(4))), T.uint32(15)))))
        for i0, i1 in T.grid(T.int64(4096), T.int64(14336)):
            with T.block("dequantize"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(compute[v_i0, v_i1], model_layers_0_mlp_down_proj_q_group_scale2[v_i0, v_i1 // T.int64(32)], model_layers_0_mlp_down_proj_q_global_scale2[T.int64(0)])
                T.writes(dequantize[v_i0, v_i1])
                dequantize[v_i0, v_i1] = compute[v_i0, v_i1] * T.Cast("float16", model_layers_0_mlp_down_proj_q_group_scale2[v_i0, v_i1 // T.int64(32)]) * T.Cast("float16", model_layers_0_mlp_down_proj_q_global_scale2[T.int64(0)])
    
    @T.prim_func
    def fused_dequantize4_NT_matmul3(model_layers_0_mlp_down_proj_q_weight7: T.Buffer((T.int64(4096), T.int64(1792)), "uint32"), model_layers_0_mlp_down_proj_q_group_scale7: T.Buffer((T.int64(4096), T.int64(448)), "float8_e4m3fn"), model_layers_0_mlp_down_proj_q_global_scale7: T.Buffer((T.int64(1),), "float32"), p_lv: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        batch_size = T.int64()
        lv = T.match_buffer(p_lv, (batch_size, T.int64(1), T.int64(14336)), "float16")
        NT_matmul_intermediate = T.match_buffer(p_output0, (batch_size, T.int64(1), T.int64(4096)), "float16")
        # with T.block("root"):
        dequantize_intermediate = T.alloc_buffer((T.int64(4096), T.int64(14336)), "float16")
        for i0, i1 in T.grid(T.int64(4096), T.int64(14336)):
            with T.block("dequantize"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(model_layers_0_mlp_down_proj_q_weight7[v_i0, v_i1 // T.int64(8)], model_layers_0_mlp_down_proj_q_group_scale7[v_i0, v_i1 // T.int64(32)], model_layers_0_mlp_down_proj_q_global_scale7[T.int64(0)])
                T.writes(dequantize_intermediate[v_i0, v_i1])
                dequantize_intermediate[v_i0, v_i1] = T.Shuffle([T.Cast("float16x2", T.reinterpret("float4_e2m1fnx2", T.Cast("uint8", T.bitwise_and(T.shift_right(model_layers_0_mlp_down_proj_q_weight7[v_i0, v_i1 // T.int64(8)], T.Cast("uint32", v_i1 % T.int64(8) // T.int64(2) * T.int64(8))), T.uint32(255)))))], [v_i1 % T.int64(2)]) * T.Cast("float16", model_layers_0_mlp_down_proj_q_group_scale7[v_i0, v_i1 // T.int64(32)]) * T.Cast("float16", model_layers_0_mlp_down_proj_q_global_scale7[T.int64(0)])
        for i0, i1, i2, k in T.grid(batch_size, T.int64(1), T.int64(4096), T.int64(14336)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(lv[v_i0, v_i1, v_k], dequantize_intermediate[v_i2, v_k])
                T.writes(NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0.0)
                NT_matmul_intermediate[v_i0, v_i1, v_i2] = NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv[v_i0, v_i1, v_k] * dequantize_intermediate[v_i2, v_k]

    @R.function
    def main(model_layers_0_mlp_down_proj_q_weight7: R.Tensor((4096, 1792), dtype="uint32"), 
             model_layers_0_mlp_down_proj_q_group_scale7: R.Tensor((4096, 448), dtype="float8_e4m3fn"),
             model_layers_0_mlp_down_proj_q_global_scale7: R.Tensor((1,), dtype="float32")):
        cls = Module
        lv810 = R.call_tir(cls.dequantize4, (model_layers_0_mlp_down_proj_q_weight7, model_layers_0_mlp_down_proj_q_group_scale7, model_layers_0_mlp_down_proj_q_global_scale7), out_sinfo=R.Tensor((4096, 14336), dtype="float16"))
        return lv810


def test_dequantize_matmul():
    tgt = tvm.target.Target("cuda")
    dev = tvm.gpu()
    mod = Module
    pipeline=relax.get_pipeline("mlc_llm", target=tgt, flashinfer=True, cublas_gemm=True, faster_transformer=True)
    with tgt:
        mod = pipeline(mod)
    ex = tvm.compile(mod, tgt)
    vm = relax.VirtualMachine(ex, dev)
    tir_build = tvm.tir.build(mod["fused_dequantize4_NT_matmul3"], tgt)
    cuda_src = tir_build.imported_modules[0].get_source()
    print(f"\n###########\ncuda source:\n {cuda_src}")


def test_dequantize():
    tgt = tvm.target.Target("cuda")
    dev = tvm.gpu()
    mod = Module
    pipeline=relax.get_pipeline("mlc_llm", target=tgt, flashinfer=True, cublas_gemm=True, faster_transformer=True)
    with tgt:
        mod = pipeline(mod)
    ex = tvm.compile(mod, tgt)
    vm = relax.VirtualMachine(ex, dev)
    tir_build = tvm.tir.build(mod["dequantize4"], tgt)
    cuda_source = tir_build.imported_modules[0].get_source()
    print(f"\n###########\ncuda source: {cuda_source}")
    # tvm.build(

    np_ipt0 = (np.random.rand(4096, 1792) * 1000).astype(np.uint32)
    np_ipt1 = (np.random.rand(4096, 448)*100).astype(float8_e4m3fn)
    np_ipt2 = (np.random.rand(1,)*100).astype(np.float32)

    ipt0 = tvm.nd.array(np_ipt0, dev)
    ipt1 = tvm.nd.array(np_ipt1, dev)
    ipt2 = tvm.nd.array(np_ipt2, dev)
    res = vm["main"](ipt0, ipt1, ipt2)
    print(f"res: {res}")
    warmup = 10
    iterations = 1000
    for _ in range(warmup):
        vm["main"](ipt0, ipt1, ipt2)

    start_t = time.time()
    for _ in range(iterations):
        vm["main"](ipt0, ipt1, ipt2)
    dur = time.time() - start_t
    
    print(f"Perf: {dur * 1000 / iterations} ms")


if __name__ == "__main__":
    test_dequantize_matmul()
