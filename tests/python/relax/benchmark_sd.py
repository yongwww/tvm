import json
import sys

sys.path.append("/home/ubuntu/octo_diffusers")
import numpy as np
import torch
import tvm
from diffusers import StableDiffusionXLPipeline, DiffusionPipeline
from octo_diffusers.unet import UNet2DConditionModel, UNetConfig
from tvm import relax
from tvm.relax.frontend.nn import spec
from tvm.relax.frontend.nn.core import set_default_dtype
from tvm.relax.backend.contrib.cutlass import partition_for_cutlass

from transformers.models.sam.clownfish_util import (
    run_opt_passes,
    offload_to_cutlass,
    run_lower_passes,
    partition_for_cutlass,
)


use_refiner = True
file_name = "/home/ubuntu/octofusion/compile/sdxl/refiner.json" if use_refiner else "base.json"
with open(file_name) as f:
    hf_config = json.load(f)

config = UNetConfig.from_dict(hf_config)

set_default_dtype("float16")
model = UNet2DConditionModel(config)


base_shape_dict = {
    "sample": spec.Tensor([2, 4, 128, 128], "float16"),
    "timestep": spec.Tensor([], "int64"),
    "encoder_hidden_states": spec.Tensor([2, 77, 2048], "float16"),
    "text_embeds": spec.Tensor([2, 1280], "float16"),
    "time_ids": spec.Tensor([2, 6], "float16"),
}
refiner_shape_dict = {
    "sample": spec.Tensor([2, 4, 128, 128], "float16"),
    "timestep": spec.Tensor([], "int64"),
    "encoder_hidden_states": spec.Tensor([2, 77, 1280], "float16"),
    "text_embeds": spec.Tensor([2, 1280], "float16"),
    "time_ids": spec.Tensor([2, 5], "float16"),
}

base = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    use_safetensors=True,
    torch_dtype=torch.float16,
    variant="fp16",
)
refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)

np.random.seed(1234)
sample = np.random.randn(2, 4, 128, 128).astype("float16")
text_embeds = np.random.randn(2, 1280).astype("float16")
if not use_refiner:
    timestep = np.asarray(981).astype("int64")
    encoder_hidden_states = np.random.randn(2, 77, 2048).astype("float16")
    time_ids = np.asarray([[1024, 1024, 0, 0, 1024, 1024], [1024, 1024, 0, 0, 1024, 1024]]).astype(
        "float16"
    )
else:
    timestep = np.asarray(151).astype("int64")
    encoder_hidden_states = np.random.randn(2, 77, 1280).astype("float16")
    time_ids = np.asarray([[1024, 1024, 0, 0, 1024], [1024, 1024, 0, 0, 1024]]).astype("float16")

print_torch = False
if print_torch:
    pipe = refiner if use_refiner else base
    pipe.to("cuda")
    torch_out = pipe.unet(
        sample=torch.tensor(sample).to("cuda"),
        timestep=torch.tensor(timestep).to("cuda"),
        encoder_hidden_states=torch.tensor(encoder_hidden_states).to("cuda"),
        added_cond_kwargs={
            "text_embeds": torch.tensor(text_embeds).to("cuda"),
            "time_ids": torch.tensor(time_ids).to("cuda"),
        },
    )
    print(torch_out)
    exit()

shape_dict = refiner_shape_dict if use_refiner else base_shape_dict
irmodule, _ = model.export_tvm(spec={"forward": shape_dict})
target = tvm.target.Target("nvidia/nvidia-a10g")

# effects = vm["_initialize_effect"]()

# Create tvm state dict from pipe weights.
if not use_refiner:
    unet_state_dict = base.unet.state_dict()
else:
    unet_state_dict = refiner.unet.state_dict()

tvm_params = {}
for k, v in unet_state_dict.items():
    tvm_params[k] = tvm.nd.array(v.numpy(), tvm.cuda())


def serialize(mod, params, prefix):
    params_dict = params

    with open("{}.json".format(prefix), "w") as fo:
        fo.write(tvm.ir.save_json(mod))

    tvm.runtime.save_param_dict_to_file(params_dict, "{}.params".format(prefix))


# serialize(irmodule, tvm_params, "unet")

tvm_params["sample"] = tvm.nd.array(sample, tvm.cuda())
tvm_params["timestep"] = tvm.nd.array(timestep, tvm.cuda())
tvm_params["encoder_hidden_states"] = tvm.nd.array(encoder_hidden_states, tvm.cuda())
tvm_params["text_embeds"] = tvm.nd.array(text_embeds, tvm.cuda())
tvm_params["time_ids"] = tvm.nd.array(time_ids, tvm.cuda())
# tvm_params[".io"] = effects


dev = tvm.gpu()

mod = tvm.ir.IRModule()
mod["main"] = irmodule["forward"]

entry_name = "main"
# apply passes
mod = run_opt_passes(mod, combine_matmul=True)


def _offload_to_cutlass(mod, target):
    # Currently, sm86 is not supported.
    sm = int(target.arch.split("_")[1])
    print("sm: ", sm)
    if sm > 80:
        sm = 80
    mod = partition_for_cutlass(mod)
    print(mod.script(show_meta=True))
    # Issues here
    mod = relax.transform.RunCodegen({"cutlass": {"sm": sm, "find_first_valid": False}})(mod)
    return mod


mod = _offload_to_cutlass(mod, target)

# print(mod.script(show_meta=True))

mod = run_lower_passes(mod, target, do_tuning=False)

exe = relax.build(mod, target=target)
vm = relax.VirtualMachine(exe, dev)

"""
with target:
    with tvm.transform.PassContext(opt_level=3):
        irmodule = relax.transform.LegalizeOps()(irmodule)
        irmodule = tvm.tir.transform.DefaultGPUSchedule()(irmodule)
        ex = relax.build(irmodule, target=target)
        vm = relax.VirtualMachine(ex, tvm.cuda())
"""

# Convert param into ordered list.
func_arity = vm._get_function_arity(entry_name)
tvm_names = [vm._get_function_param_name(entry_name, i) for i in range(func_arity)]
input_args = [tvm_params[name] for name in tvm_names]
tvm_output = vm[entry_name](*input_args)

print("tvm_outputL ", tvm_output)

# breakpoint()
