import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T
import tvm.testing


@T.prim_func
def fused_NT_matmul1_multiply_multiply4_cast7(astype132: T.Buffer((T.int64(1), T.int64(1), T.int64(2048)), "e4m3_float8"), model_layers_0_self_attn_o_proj_q_weight3: T.Buffer((T.int64(2048), T.int64(2048)), "e4m3_float8"), model_layers_0_self_attn_o_proj_q_calibration_scale3: T.Buffer((T.int64(1),), "float32"), model_layers_0_self_attn_o_proj_q_scale3: T.Buffer((T.int64(1),), "float32"), compute_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(2048)), "float16")):
    blockIdx_x = T.launch_thread("blockIdx.x", 128)
    astype132_shared = T.allocate([512], "e4m3_float8x4", "shared")
    NT_matmul_rf_local = T.allocate([4], "float32", "local")
    model_layers_0_self_attn_o_proj_q_weight3_local = T.allocate([2], "e4m3_float8x4", "local")
    NT_matmul_rf_local_1 = T.allocate([1], "float32", "local")
    cross_thread_NT_matmul_intermediate_local = T.allocate([1], "float32", "local")
    threadIdx_x = T.launch_thread("threadIdx.x", 32)
    threadIdx_y = T.launch_thread("threadIdx.y", 16)
    ax2_0 = T.int64()
    astype132_shared_1 = T.Buffer((T.int64(512),), "e4m3_float8x4", data=astype132_shared, scope="shared")
    # with T.attr(ax2_0, "pragma_vectorize", 1):
    for ax2_0_yongwww in range(T.int64(1)):
        astype132_1 = T.Buffer((T.int64(2048),), "e4m3_float8", data=astype132.data)
        astype132_shared_1[threadIdx_y * 32 + threadIdx_x] = astype132_1[threadIdx_y * 128 + threadIdx_x * 4:threadIdx_y * 128 + threadIdx_x * 4 + 4]
    NT_matmul_rf_local_2 = T.Buffer((T.int64(4),), data=NT_matmul_rf_local, scope="local")
    NT_matmul_rf_local_2[0:4] = T.Broadcast(T.float32(0.0), 4)
    model_layers_0_self_attn_o_proj_q_weight3_local_1 = T.Buffer((T.int64(2),), "e4m3_float8x4", data=model_layers_0_self_attn_o_proj_q_weight3_local, scope="local")
    model_layers_0_self_attn_o_proj_q_weight3_1 = T.Buffer((T.int64(4194304),), "e4m3_float8", data=model_layers_0_self_attn_o_proj_q_weight3.data)
    model_layers_0_self_attn_o_proj_q_weight3_local_1[0] = model_layers_0_self_attn_o_proj_q_weight3_1[blockIdx_x * 32768 + threadIdx_y * 2048 + threadIdx_x * 8:blockIdx_x * 32768 + threadIdx_y * 2048 + threadIdx_x * 8 + 4]
    model_layers_0_self_attn_o_proj_q_weight3_local_1[1] = model_layers_0_self_attn_o_proj_q_weight3_1[blockIdx_x * 32768 + threadIdx_y * 2048 + threadIdx_x * 8 + 4:blockIdx_x * 32768 + threadIdx_y * 2048 + threadIdx_x * 8 + 4 + 4]
    NT_matmul_rf_local_2[0:4] = NT_matmul_rf_local_2[0:4] + T.Cast("float32x4", astype132_shared_1[threadIdx_x * 2]) * T.Cast("float32x4", model_layers_0_self_attn_o_proj_q_weight3_local_1[0])
    NT_matmul_rf_local_2[0:4] = NT_matmul_rf_local_2[0:4] + T.Cast("float32x4", astype132_shared_1[threadIdx_x * 2 + 1]) * T.Cast("float32x4", model_layers_0_self_attn_o_proj_q_weight3_local_1[1])
    model_layers_0_self_attn_o_proj_q_weight3_local_1[0] = model_layers_0_self_attn_o_proj_q_weight3_1[blockIdx_x * 32768 + threadIdx_y * 2048 + threadIdx_x * 8 + 256:blockIdx_x * 32768 + threadIdx_y * 2048 + threadIdx_x * 8 + 256 + 4]
    model_layers_0_self_attn_o_proj_q_weight3_local_1[1] = model_layers_0_self_attn_o_proj_q_weight3_1[blockIdx_x * 32768 + threadIdx_y * 2048 + threadIdx_x * 8 + 260:blockIdx_x * 32768 + threadIdx_y * 2048 + threadIdx_x * 8 + 260 + 4]
    NT_matmul_rf_local_2[0:4] = NT_matmul_rf_local_2[0:4] + T.Cast("float32x4", astype132_shared_1[threadIdx_x * 2 + 64]) * T.Cast("float32x4", model_layers_0_self_attn_o_proj_q_weight3_local_1[0])
    NT_matmul_rf_local_2[0:4] = NT_matmul_rf_local_2[0:4] + T.Cast("float32x4", astype132_shared_1[threadIdx_x * 2 + 65]) * T.Cast("float32x4", model_layers_0_self_attn_o_proj_q_weight3_local_1[1])
    model_layers_0_self_attn_o_proj_q_weight3_local_1[0] = model_layers_0_self_attn_o_proj_q_weight3_1[blockIdx_x * 32768 + threadIdx_y * 2048 + threadIdx_x * 8 + 512:blockIdx_x * 32768 + threadIdx_y * 2048 + threadIdx_x * 8 + 512 + 4]
    model_layers_0_self_attn_o_proj_q_weight3_local_1[1] = model_layers_0_self_attn_o_proj_q_weight3_1[blockIdx_x * 32768 + threadIdx_y * 2048 + threadIdx_x * 8 + 516:blockIdx_x * 32768 + threadIdx_y * 2048 + threadIdx_x * 8 + 516 + 4]
    NT_matmul_rf_local_2[0:4] = NT_matmul_rf_local_2[0:4] + T.Cast("float32x4", astype132_shared_1[threadIdx_x * 2 + 128]) * T.Cast("float32x4", model_layers_0_self_attn_o_proj_q_weight3_local_1[0])
    NT_matmul_rf_local_2[0:4] = NT_matmul_rf_local_2[0:4] + T.Cast("float32x4", astype132_shared_1[threadIdx_x * 2 + 129]) * T.Cast("float32x4", model_layers_0_self_attn_o_proj_q_weight3_local_1[1])
    model_layers_0_self_attn_o_proj_q_weight3_local_1[0] = model_layers_0_self_attn_o_proj_q_weight3_1[blockIdx_x * 32768 + threadIdx_y * 2048 + threadIdx_x * 8 + 768:blockIdx_x * 32768 + threadIdx_y * 2048 + threadIdx_x * 8 + 768 + 4]
    model_layers_0_self_attn_o_proj_q_weight3_local_1[1] = model_layers_0_self_attn_o_proj_q_weight3_1[blockIdx_x * 32768 + threadIdx_y * 2048 + threadIdx_x * 8 + 772:blockIdx_x * 32768 + threadIdx_y * 2048 + threadIdx_x * 8 + 772 + 4]
    NT_matmul_rf_local_2[0:4] = NT_matmul_rf_local_2[0:4] + T.Cast("float32x4", astype132_shared_1[threadIdx_x * 2 + 192]) * T.Cast("float32x4", model_layers_0_self_attn_o_proj_q_weight3_local_1[0])
    NT_matmul_rf_local_2[0:4] = NT_matmul_rf_local_2[0:4] + T.Cast("float32x4", astype132_shared_1[threadIdx_x * 2 + 193]) * T.Cast("float32x4", model_layers_0_self_attn_o_proj_q_weight3_local_1[1])
    model_layers_0_self_attn_o_proj_q_weight3_local_1[0] = model_layers_0_self_attn_o_proj_q_weight3_1[blockIdx_x * 32768 + threadIdx_y * 2048 + threadIdx_x * 8 + 1024:blockIdx_x * 32768 + threadIdx_y * 2048 + threadIdx_x * 8 + 1024 + 4]
    model_layers_0_self_attn_o_proj_q_weight3_local_1[1] = model_layers_0_self_attn_o_proj_q_weight3_1[blockIdx_x * 32768 + threadIdx_y * 2048 + threadIdx_x * 8 + 1028:blockIdx_x * 32768 + threadIdx_y * 2048 + threadIdx_x * 8 + 1028 + 4]
    NT_matmul_rf_local_2[0:4] = NT_matmul_rf_local_2[0:4] + T.Cast("float32x4", astype132_shared_1[threadIdx_x * 2 + 256]) * T.Cast("float32x4", model_layers_0_self_attn_o_proj_q_weight3_local_1[0])
    NT_matmul_rf_local_2[0:4] = NT_matmul_rf_local_2[0:4] + T.Cast("float32x4", astype132_shared_1[threadIdx_x * 2 + 257]) * T.Cast("float32x4", model_layers_0_self_attn_o_proj_q_weight3_local_1[1])
    model_layers_0_self_attn_o_proj_q_weight3_local_1[0] = model_layers_0_self_attn_o_proj_q_weight3_1[blockIdx_x * 32768 + threadIdx_y * 2048 + threadIdx_x * 8 + 1280:blockIdx_x * 32768 + threadIdx_y * 2048 + threadIdx_x * 8 + 1280 + 4]
    model_layers_0_self_attn_o_proj_q_weight3_local_1[1] = model_layers_0_self_attn_o_proj_q_weight3_1[blockIdx_x * 32768 + threadIdx_y * 2048 + threadIdx_x * 8 + 1284:blockIdx_x * 32768 + threadIdx_y * 2048 + threadIdx_x * 8 + 1284 + 4]
    NT_matmul_rf_local_2[0:4] = NT_matmul_rf_local_2[0:4] + T.Cast("float32x4", astype132_shared_1[threadIdx_x * 2 + 320]) * T.Cast("float32x4", model_layers_0_self_attn_o_proj_q_weight3_local_1[0])
    NT_matmul_rf_local_2[0:4] = NT_matmul_rf_local_2[0:4] + T.Cast("float32x4", astype132_shared_1[threadIdx_x * 2 + 321]) * T.Cast("float32x4", model_layers_0_self_attn_o_proj_q_weight3_local_1[1])
    model_layers_0_self_attn_o_proj_q_weight3_local_1[0] = model_layers_0_self_attn_o_proj_q_weight3_1[blockIdx_x * 32768 + threadIdx_y * 2048 + threadIdx_x * 8 + 1536:blockIdx_x * 32768 + threadIdx_y * 2048 + threadIdx_x * 8 + 1536 + 4]
    model_layers_0_self_attn_o_proj_q_weight3_local_1[1] = model_layers_0_self_attn_o_proj_q_weight3_1[blockIdx_x * 32768 + threadIdx_y * 2048 + threadIdx_x * 8 + 1540:blockIdx_x * 32768 + threadIdx_y * 2048 + threadIdx_x * 8 + 1540 + 4]
    NT_matmul_rf_local_2[0:4] = NT_matmul_rf_local_2[0:4] + T.Cast("float32x4", astype132_shared_1[threadIdx_x * 2 + 384]) * T.Cast("float32x4", model_layers_0_self_attn_o_proj_q_weight3_local_1[0])
    NT_matmul_rf_local_2[0:4] = NT_matmul_rf_local_2[0:4] + T.Cast("float32x4", astype132_shared_1[threadIdx_x * 2 + 385]) * T.Cast("float32x4", model_layers_0_self_attn_o_proj_q_weight3_local_1[1])
    model_layers_0_self_attn_o_proj_q_weight3_local_1[0] = model_layers_0_self_attn_o_proj_q_weight3_1[blockIdx_x * 32768 + threadIdx_y * 2048 + threadIdx_x * 8 + 1792:blockIdx_x * 32768 + threadIdx_y * 2048 + threadIdx_x * 8 + 1792 + 4]
    model_layers_0_self_attn_o_proj_q_weight3_local_1[1] = model_layers_0_self_attn_o_proj_q_weight3_1[blockIdx_x * 32768 + threadIdx_y * 2048 + threadIdx_x * 8 + 1796:blockIdx_x * 32768 + threadIdx_y * 2048 + threadIdx_x * 8 + 1796 + 4]
    NT_matmul_rf_local_2[0:4] = NT_matmul_rf_local_2[0:4] + T.Cast("float32x4", astype132_shared_1[threadIdx_x * 2 + 448]) * T.Cast("float32x4", model_layers_0_self_attn_o_proj_q_weight3_local_1[0])
    NT_matmul_rf_local_2[0:4] = NT_matmul_rf_local_2[0:4] + T.Cast("float32x4", astype132_shared_1[threadIdx_x * 2 + 449]) * T.Cast("float32x4", model_layers_0_self_attn_o_proj_q_weight3_local_1[1])
    NT_matmul_rf_local_3 = T.Buffer((T.int64(1),), data=NT_matmul_rf_local_1, scope="local")
    NT_matmul_rf_local_3[0] = T.float32(0.0)
    NT_matmul_rf_local_3[0] = NT_matmul_rf_local_3[0] + NT_matmul_rf_local_2[0]
    NT_matmul_rf_local_3[0] = NT_matmul_rf_local_3[0] + NT_matmul_rf_local_2[1]
    NT_matmul_rf_local_3[0] = NT_matmul_rf_local_3[0] + NT_matmul_rf_local_2[2]
    NT_matmul_rf_local_3[0] = NT_matmul_rf_local_3[0] + NT_matmul_rf_local_2[3]
    with T.attr(T.comm_reducer(lambda x0, y0: x0 + y0, [T.float32(0.0)]), "reduce_scope", T.reinterpret("handle", T.uint64(0))):
        cross_thread_NT_matmul_intermediate_local_1 = T.Buffer((1,), data=cross_thread_NT_matmul_intermediate_local, scope="local")
        T.tvm_thread_allreduce(T.uint32(1), NT_matmul_rf_local_3[0], T.bool(True), cross_thread_NT_matmul_intermediate_local_1[0], threadIdx_x)
    if threadIdx_x == 0:
        compute_intermediate_1 = T.Buffer((T.int64(2048),), "float16", data=compute_intermediate.data)
        cross_thread_NT_matmul_intermediate_local_1 = T.Buffer((T.int64(1),), data=cross_thread_NT_matmul_intermediate_local, scope="local")
        model_layers_0_self_attn_o_proj_q_calibration_scale3_1 = T.Buffer((T.int64(1),), data=model_layers_0_self_attn_o_proj_q_calibration_scale3.data)
        model_layers_0_self_attn_o_proj_q_scale3_1 = T.Buffer((T.int64(1),), data=model_layers_0_self_attn_o_proj_q_scale3.data)
        compute_intermediate_1[blockIdx_x * 16 + threadIdx_y] = T.Cast("float16", cross_thread_NT_matmul_intermediate_local_1[0] * (model_layers_0_self_attn_o_proj_q_calibration_scale3_1[0] * model_layers_0_self_attn_o_proj_q_scale3_1[0]))

def test_tir():
    target = tvm.target.Target("nvidia/nvidia-h100")
    # Failed if target is a100 or others
    mod = tvm.build(fused_NT_matmul1_multiply_multiply4_cast7, target=target)

if __name__ == "__main__":
    test_tir()
