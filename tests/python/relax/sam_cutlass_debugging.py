import numpy as np

import tvm
from tvm import relax, tir
from tvm.script import ir as I
from tvm.script import relax as R, tir as T
import tvm.testing
import tvm.topi.testing

from tvm.relax.backend.contrib.cutlass import partition_for_cutlass



def offload_to_cutlass(
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



@I.ir_module
class Module:
    @R.function
    def main(
        input_points: R.Tensor((1, 1, 2, 2), dtype="float32"),
        prompt_encoder_shared_embedding_positional_embedding: R.Tensor((2, 128), dtype="float32"),
    ):  # -> R.Tuple(R.Tensor((1, 1, 2, 128), dtype="float32"), R.Tensor((1, 1, 2, 128), dtype="float32")):
        R.func_attr({"num_input": 2})
        with R.dataflow():
            matmul146: R.Tensor((1, 1, 2, 128), dtype="float32") = R.matmul(
                input_points, prompt_encoder_shared_embedding_positional_embedding, out_dtype="void"
            )
            mul128: R.Tensor((1, 1, 2, 128), dtype="float32") = R.multiply(
                R.const(6.2831854820251465, "float32"), matmul146
                # R.const(1., "float32"), matmul146
            )
            debugging: R.Tensor((1, 1, 2, 128), dtype="float32") = R.cos(mul128)
            R.output(matmul146, debugging)
        return (matmul146, debugging)


def get_random_inputs(max_val=1):
    np.random.seed(1)
    input_points = tvm.nd.array(np.random.rand(1, 1, 2, 2).astype(np.float32), tvm.gpu())
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

    if apply_cutlass:
        mod = offload_to_cutlass(mod, target, ["main"])
    else:
        # print("Module with R.cos w/o cutlass: \n", mod.script(show_meta=True))
        pass

    # print(mod.script(show_meta=True))

    mod = run_lower_passes(mod, target, do_tuning=False)

    # print(mod.script(show_meta=True))

    exe = relax.build(mod, target=target)
    # print("Sam with cos exe (no cutlass): \n", exe.as_text())
    vm = relax.VirtualMachine(exe, dev)

    return vm[entry_name](*inputs)


if __name__ == "__main__":
    # inputs = get_random_inputs(max_val=1)
    inputs = get_random_inputs(max_val=1)

    out2 = test_matmul_cutlass(inputs, apply_cutlass=False)
    out1 = test_matmul_cutlass(inputs, apply_cutlass=True)

    if isinstance(out1, tvm.container.Array):
        for i, (o1, o2) in enumerate(zip(out1, out2)):
            print('Comparing output ', i)
            tvm.testing.assert_allclose(o1.numpy(), o2.numpy(), rtol=1e-1, atol=1e-1)
    else:
        tvm.testing.assert_allclose(out1.numpy(), out2.numpy(), rtol=1e-1, atol=1e-1)

    print("Inference results match!")
