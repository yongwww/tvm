import numpy as np
import torch

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
class TVMModule:
    @R.function
    def main(subtract2: R.Tensor((64, 64, 2), dtype="float16"), # [-1, 1], like -0.9844,  0.9844
             prompt_encoder_shared_embedding_positional_embedding: R.Tensor((2, 128), dtype="float16")): # [-1000, 1000.0], norm distri
        R.func_attr({"global_symbol": "main", "num_input": 2})
        with R.dataflow():
            matmul: R.Tensor((64, 64, 128), dtype="float16") = R.matmul(subtract2, prompt_encoder_shared_embedding_positional_embedding, out_dtype="void")
            mul: R.Tensor((64, 64, 128), dtype="float16") = R.multiply(R.const(6.28125, "float16"), matmul)
            sin: R.Tensor((64, 64, 128), dtype="float16") = R.sin(mul)
            cos: R.Tensor((64, 64, 128), dtype="float16") = R.cos(mul)
            gv: R.Tensor((64, 64, 256), dtype="float16") = R.concat((sin, cos), axis=2)
            R.output(gv)
        return gv

torch.backends.cuda.matmul.allow_tf32 = False
class PTModel(torch.nn.Module):
    def __init__(self):
        super(PTModel, self).__init__()
        # Any additional initializations can be added here

    def forward(self, coordinates, positional_embedding):
        # Ensure the inputs are in FP16
        coordinates = coordinates.to(torch.float16)
        positional_embedding = positional_embedding.to(torch.float16)

        # Main logic
        coordinates = torch.matmul(coordinates, positional_embedding)
        coordinates = 2 * np.pi * coordinates

        # Concatenating sin and cos transformations
        return torch.cat([torch.sin(coordinates), torch.cos(coordinates)], dim=-1)


def np_model(coordinates, positional_embedding):
    # Ensure the inputs are in FP16
    # coordinates = coordinates.astype(np.float16)
    # positional_embedding = positional_embedding.astype(np.float16)

    # Main logic
    coordinates = np.matmul(coordinates, positional_embedding)
    coordinates = 2 * np.pi * coordinates

    # Concatenating sin and cos transformations
    sin_coordinates = np.sin(coordinates)
    cos_coordinates = np.cos(coordinates)
    return np.concatenate([sin_coordinates, cos_coordinates], axis=-1)


def jax_model(coordinates, positional_embedding):
    import jax
    import jax.numpy as jnp
    # Convert the inputs to JAX arrays
    coordinates = jnp.array(coordinates)
    positional_embedding = jnp.array(positional_embedding)

    # Ensure the inputs are in FP16
    coordinates = coordinates.astype(jnp.float16)
    positional_embedding = positional_embedding.astype(jnp.float16)

    # Main logic
    coordinates = jnp.matmul(coordinates, positional_embedding)
    coordinates = 2 * jnp.pi * coordinates

    # Concatenating sin and cos transformations
    sin_coordinates = jnp.sin(coordinates)
    cos_coordinates = jnp.cos(coordinates)
    return jnp.concatenate([sin_coordinates, cos_coordinates], axis=-1)


def get_numpy_inputs(max_val=1):
    np.random.seed(1)
    ipt0 = np.random.rand(64, 64, 2).astype(np.float16)
    # tvm.nd.array(np.random.rand(64, 64, 2).astype(np.float16), tvm.gpu())
    ipt1 = np.random.uniform(low=-max_val, high=max_val, size=(2, 128)).astype(np.float16)
    return [ipt0, ipt1]


def compile_sam(apply_cutlass=True):
    entry_name = "main"
    dtype = "float32"
    target, dev = tvm.target.Target("nvidia/nvidia-a100"), tvm.gpu()

    mod = TVMModule

    if apply_cutlass:
        mod = offload_to_cutlass(mod, target, [entry_name])
    else:
        # print("Module with R.cos w/o cutlass: \n", mod.script(show_meta=True))
        pass

    # print(mod.script(show_meta=True))

    mod = run_lower_passes(mod, target, do_tuning=False)

    # print(mod.script(show_meta=True))

    exe = relax.build(mod, target=target)
    # print("Sam with cos exe: \n", exe.as_text())
    # print(
    #    "cuda code: ",
    #    exe.mod.imported_modules[0].imported_modules[0].imported_modules[0].get_source(),
    # )
    return relax.VirtualMachine(exe, dev)


def test_correctness():
    # inputs = get_random_inputs(max_val=1)
    entry_name = "main"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    np_inputs = get_numpy_inputs(max_val=1000)
    np_inputs_fp32 = [input.astype(np.float32) for input in np_inputs]
    tvm_inputs = [tvm.nd.array(input, tvm.gpu()) for input in np_inputs]
    pt_inputs = [torch.from_numpy(input).to(device) for input in np_inputs]
    
    vm_wo_cutlass = compile_sam(apply_cutlass=False)

    tvm_out = vm_wo_cutlass[entry_name](*tvm_inputs)
    pt_mod = PTModel().to(device)
    pt_out = pt_mod(*pt_inputs)

    np_out = np_model(*np_inputs)
    np_out_fp32 = np_model(*np_inputs_fp32)
    jax_out = jax_model(*np_inputs)
    jax_out_fp32 = jax_model(*np_inputs_fp32)

    print("np_inputs dtype: ", np_inputs[0].dtype)
    print("np_inputs_fp32 dtype: ", np_inputs_fp32[0].dtype)

    print("np_out: \n", np_out)
    print("np_out_fp32, dtype: : ", np_out_fp32.dtype, "\nvalue: \n: ", np_out_fp32)
    print("jax_out: \n", jax_out)
    print("jax_out_fp32: \n", jax_out_fp32)

    if isinstance(tvm_out, tvm.container.Array):
        for i, (o1, o2) in enumerate(zip(tvm_out, pt_out)):
            print("Comparing output ", i)
            tvm.testing.assert_allclose(o1.numpy(), o2.cpu().detach().numpy(), rtol=1e-1, atol=1e-1)
    else:
        tvm.testing.assert_allclose(
            tvm_out.numpy(), pt_out.cpu().detach().numpy(), rtol=1e-1, atol=1e-1
        )

    print("Inference results match!")


if __name__ == "__main__":
    test_correctness()
    # test_pt()
    # lib_compare()
