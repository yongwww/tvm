import numpy as np

import tvm
from tvm import relax, tir

from tvm.script import ir as I
from tvm.script import relax as R, tir as T

import tvm.testing
import tvm.topi.testing

from tvm.relax.backend.contrib.cutlass import partition_for_cutlass


def run_opt_passes(mod, params=None, fp16_input_names=None, combine_matmul=False):
    passes = [
        relax.transform.EliminateCommonSubexpr(),
        relax.transform.CanonicalizeBindings(),
        # relax.transform.ConvertLayout({"relax.nn.conv2d": ["NHWC", "OHWI"]}),
        # get_rewrite_pass(combine_matmul)
        relax.transform.DeadCodeElimination(["main"]),
    ]

    if params:
        passes += [
            relax.transform.BindParams("main", params),
            relax.transform.FoldConstant(),
            relax.transform.ToMixedPrecision(out_dtype="float16"),
        ]
    else:
        passes += [
            relax.transform.FoldConstant(),
            # relax.transform.ToMixedPrecision(
            #     out_dtype="float16", fp16_input_names=fp16_input_names
            # ),
        ]
        """
              File "/home/ubuntu/tvm/src/relax/transform/infer_amp_utils.cc", line 28
              InternalError: Check failed: (tensor) is false: Expected TensorStructInfo, but got R.Objec
        """,

    return tvm.transform.Sequential(passes)(mod)


def run_lower_passes(mod, target, do_tuning=True):
    passes = [relax.pipeline.get_pipeline()]

    if "cuda" in target.kind.name:
        work_dir = "logs"
        with target:
            if do_tuning:
                passes.append(
                    relax.transform.MetaScheduleTuneIRMod(
                        params={},
                        work_dir=work_dir,
                        max_trials_global=1400,
                        max_trials_per_task=50,
                    )
                )
            # passes.append(relax.transform.MetaScheduleApplyDatabase(work_dir))
            passes.append(tir.transform.DefaultGPUSchedule())

    with target, tvm.transform.PassContext(opt_level=3):
        return tvm.transform.Sequential(passes)(mod)


metadata = tvm.ir.load_json(
    """{
  \"root\": 1, 
  \"nodes\": [
    {
      \"type_key\": \"\"
    }, 
    {
      \"type_key\": \"Map\", 
      \"keys\": [
        \"relax.expr.Constant\"
      ], 
      \"data\": [2]
    }, 
    {
      \"type_key\": \"Array\", 
      \"data\": [3]
    }, 
    {
      \"type_key\": \"relax.expr.Constant\", 
      \"attrs\": {
        \"_checked_type_\": \"13\", 
        \"data\": \"0\", 
        \"span\": \"0\", 
        \"struct_info_\": \"4\"
      }
    }, 
    {
      \"type_key\": \"relax.TensorStructInfo\", 
      \"attrs\": {
        \"dtype\": \"float32\", 
        \"ndim\": \"4\", 
        \"shape\": \"5\", 
        \"span\": \"0\", 
        \"vdevice\": \"0\"
      }
    }, 
    {
      \"type_key\": \"relax.expr.ShapeExpr\", 
      \"attrs\": {
        \"_checked_type_\": \"12\", 
        \"span\": \"0\", 
        \"struct_info_\": \"11\", 
        \"values\": \"6\"
      }
    }, 
    {
      \"type_key\": \"Array\", 
      \"data\": [7, 8, 9, 10]
    }, 
    {
      \"type_key\": \"IntImm\", 
      \"attrs\": {
        \"dtype\": \"int64\", 
        \"span\": \"0\", 
        \"value\": \"1\"
      }
    }, 
    {
      \"type_key\": \"IntImm\", 
      \"attrs\": {
        \"dtype\": \"int64\", 
        \"span\": \"0\", 
        \"value\": \"1\"
      }
    }, 
    {
      \"type_key\": \"IntImm\", 
      \"attrs\": {
        \"dtype\": \"int64\", 
        \"span\": \"0\", 
        \"value\": \"1\"
      }
    }, 
    {
      \"type_key\": \"IntImm\", 
      \"attrs\": {
        \"dtype\": \"int64\", 
        \"span\": \"0\", 
        \"value\": \"2\"
      }
    }, 
    {
      \"type_key\": \"relax.ShapeStructInfo\", 
      \"attrs\": {
        \"ndim\": \"4\", 
        \"span\": \"0\", 
        \"values\": \"6\"
      }
    }, 
    {
      \"type_key\": \"relax.ShapeType\", 
      \"attrs\": {
        \"ndim\": \"4\", 
        \"span\": \"0\"
      }
    }, 
    {
      \"type_key\": \"relax.DynTensorType\", 
      \"attrs\": {
        \"dtype\": \"float32\", 
        \"ndim\": \"4\", 
        \"span\": \"0\"
      }
    }
  ], 
  \"b64ndarrays\": [
    \"P6G0lvBAXt0AAAAAAAAAAAEAAAAAAAAABAAAAAIgAQABAAAAAAAAAAEAAAAAAAAAAQAAAAAAAAACAAAAAAAAAAgAAAAAAAAAAAAAAAAAAAA=\"
  ], 
  \"attrs\": {\"tvm_version\": \"0.15.dev0\"}
}"""
)


@I.ir_module
class Module:
    @R.function
    def main(
        input_points: R.Tensor((1, 1, 1, 2), dtype="float32"),
        prompt_encoder_shared_embedding_positional_embedding: R.Tensor((2, 128), dtype="float32"),
    ):  # -> R.Tuple(R.Tensor((1, 1, 2, 128), dtype="float32"), R.Tensor((1, 1, 2, 128), dtype="float32")):
        R.func_attr({"num_input": 3})
        with R.dataflow():
            add255: R.Tensor((1, 1, 1, 2), dtype="float32") = R.add(
                input_points, R.const(0.5, "float32")
            )
            concat6: R.Tensor((1, 1, 2, 2), dtype="float32") = R.concat(
                (add255, metadata["relax.expr.Constant"][0]), axis=2
            )
            strided_slice18: R.Tensor((1, 1, 2, 1), dtype="float32") = R.strided_slice(
                concat6, axes=[3], begin=[0], end=[1], strides=None, assume_inbound=False
            )
            divide8: R.Tensor((1, 1, 2, 1), dtype="float32") = R.divide(
                strided_slice18, R.const(1024, "float32")
            )
            strided_slice19: R.Tensor((1, 1, 2, 1), dtype="float32") = R.strided_slice(
                concat6, axes=[3], begin=[1], end=[2], strides=None, assume_inbound=False
            )
            divide9: R.Tensor((1, 1, 2, 1), dtype="float32") = R.divide(
                strided_slice19, R.const(1024, "float32")
            )
            concat8: R.Tensor((1, 1, 2, 2), dtype="float32") = R.concat((divide8, divide9), axis=3)
            add256: R.Tensor((1, 1, 2, 2), dtype="float32") = R.add(concat8, concat8)
            subtract56: R.Tensor((1, 1, 2, 2), dtype="float32") = R.subtract(
                add256, R.const(1, "float32")
            )
            matmul146: R.Tensor((1, 1, 2, 128), dtype="float32") = R.matmul(
                subtract56, prompt_encoder_shared_embedding_positional_embedding, out_dtype="void"
            )
            mul128: R.Tensor((1, 1, 2, 128), dtype="float32") = R.multiply(
                R.const(6.2831854820251465, "float32"), matmul146
            )
            debugging: R.Tensor((1, 1, 2, 128), dtype="float32") = R.cos(mul128)
            R.output(matmul146, debugging)
        return (matmul146, debugging)


def _offload_to_cutlass(
    mod, target, entry_functions=["main", "get_prompt_embeddings", "get_image_embeddings"]
):
    # Currently, sm86 is not supported.
    sm = int(target.arch.split("_")[1])
    print("sm: ", sm)
    if sm > 80:
        sm = 80
    mod = partition_for_cutlass(mod)

    # print("Module with R.cos after cutlass partition: \n", mod.script(show_meta=True))
    # mod.show()
    mod = relax.transform.RunCodegen(
        {"cutlass": {"sm": sm, "find_first_valid": False}},
        entry_functions=entry_functions,
    )(mod)
    # print("Module with R.sqrt after cutlass RunCodegen: \n", mod.script(show_meta=True))

    return mod


def get_random_inputs(max_val=1):
    input_points = tvm.nd.array(np.random.rand(1, 1, 1, 2).astype(np.float32), tvm.gpu())
    embedding = tvm.nd.array(
        np.random.uniform(low=-max_val, high=max_val, size=(2, 128)).astype(np.float32), tvm.gpu()
    )
    inputs = [input_points, embedding]
    return inputs


def test_matmul_cutlass(inputs=None, apply_cutlass=True):
    entry_name = "main"
    dtype = "float32"
    target, dev = tvm.target.Target("nvidia/nvidia-a100"), tvm.gpu()

    mod = Module

    mod = run_opt_passes(mod, combine_matmul=False)
    # print(mod.script(show_meta=True))
    if apply_cutlass:
        mod = _offload_to_cutlass(mod, target, ["main"])
    else:
        # print("Module with R.cos w/o cutlass: \n", mod.script(show_meta=True))
        pass

    # print(mod.script(show_meta=True))

    mod = run_lower_passes(mod, target, do_tuning=False)

    # print(mod.script(show_meta=True))

    exe = relax.build(mod, target=target)
    # print("Sam with cos exe (no cutlass): \n", exe.as_text())
    vm = relax.VirtualMachine(exe, dev)

    # vm_profiler = relax.VirtualMachine(exe, tvm.gpu(), profile=True)
    # report = vm_profiler.profile(entry_name, *input_args)

    return vm[entry_name](*inputs)


if __name__ == "__main__":
    inputs = get_random_inputs(max_val=1000)

    out2 = test_matmul_cutlass(inputs, apply_cutlass=False)
    out1 = test_matmul_cutlass(inputs, apply_cutlass=True)

    tvm.testing.assert_allclose(out1[0].asnumpy(), out2[0].asnumpy(), rtol=1e-1, atol=1e-1)
    tvm.testing.assert_allclose(out1[1].asnumpy(), out2[1].asnumpy(), rtol=1e-1, atol=1e-1)

    print("Inference results match!")
