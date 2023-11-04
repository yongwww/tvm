import torch
from PIL import Image
import requests
import time

# from transformers import SamModel, SamProcessor

from transformers.models.sam.modeling_sam import SamModel as PTSamModel
from transformers.models.sam.processing_sam import SamProcessor

import tvm
from tvm import relax, tir
from tvm.relax.frontend.nn import spec
import numpy as np

import tvm.topi.testing
from tvm.relax.transform import LegalizeOps
from tvm.script import relax as R, tir as T
import tvm.testing

from transformers.models.sam.configuration_sam import (
    SamConfig,
    SamVisionConfig,
    SamPromptEncoderConfig,
    SamMaskDecoderConfig,
)

from transformers.models.sam.modeling_tvm_sam import SamModel as RXSamModel

from transformers.models.sam.clownfish_util import (
    run_opt_passes,
    offload_to_cutlass,
    run_lower_passes,
    partition_for_cutlass,
)


def get_inputs():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
    input_points = [[[450, 600]]]  # 2D location of a window in the image

    inputs = processor(raw_image, input_points=input_points, return_tensors="pt").to(device)
    # print("inputs: ", inputs)
    return inputs


device = "cuda" if torch.cuda.is_available() else "cpu"
config = SamConfig()
pt_sam_model = PTSamModel(config).to(device)


def test_tvm_sam():
    batch_size, total_seq_len, dtype = 1, 32, "float32"

    mod_spec = {
        "get_image_embeddings": {
            # (batch_size, num_channels, height, width)
            "pixel_values": spec.Tensor([1, 3, 1024, 1024], "float32"),
        },
        "get_prompt_embeddings": {"input_points": spec.Tensor([1, 1, 1, 2], "float32")},
        "forward": {
            "pixel_values": spec.Tensor([1, 3, 1024, 1024], "float32"),
            "input_points": spec.Tensor([1, 1, 1, 2], "float32"),
        },
    }

    config = SamConfig()
    relax_model = RXSamModel(config)
    ir_mod, _ = relax_model.export_tvm(spec=mod_spec, debug=True)

    mod = tvm.ir.IRModule()
    mod["main"] = ir_mod["forward"]
    mod["get_prompt_embeddings"] = ir_mod["get_prompt_embeddings"]
    mod["get_image_embeddings"] = ir_mod["get_image_embeddings"]

    mod["_initialize_effect"] = ir_mod["_initialize_effect"]
    entry_name = "main"

    # target, dev = "llvm", tvm.cpu()
    target = tvm.target.Target("nvidia/nvidia-a10g")  # tvm.target.Target("cuda", host="llvm")
    dev = tvm.gpu()

    # apply passes
    mod = run_opt_passes(mod, combine_matmul=True)
    # mod = offload_to_cutlass(mod, target)

    # Apply cutlass optimization.
    # mod = partition_for_cutlass(mod)
    # mod = relax.transform.RunCodegen(
    #    {"cutlass": {"sm": 80, "find_first_valid": False}}
    # )(mod)

    mod = run_lower_passes(mod, target, do_tuning=False)

    # with tvm.target.Target("cuda"):
    #     mod = tvm.tir.transform.DefaultGPUSchedule()(mod)

    exe = relax.build(mod, target=target)
    vm = relax.VirtualMachine(exe, dev)

    # Prepare inputs for inference
    global pt_sam_model
    tvm_params = {}
    for k, v in pt_sam_model.state_dict().items():
        tvm_params[k] = tvm.nd.array(v.cpu().numpy(), dev)

    # image input
    img_inputs = get_inputs()
    for k, v in img_inputs.items():
        tvm_params[k] = tvm.nd.array(v.cpu().float().numpy(), dev)

    effects = vm["_initialize_effect"]()
    tvm_params[".io"] = effects

    # Convert param into ordered list.
    func_arity = vm._get_function_arity(entry_name)
    tvm_names = [vm._get_function_param_name(entry_name, i) for i in range(func_arity)]

    input_args = [tvm_params[name] for name in tvm_names]
    # vm_profiler = relax.VirtualMachine(exe, tvm.gpu(), profile=True)
    # report = vm_profiler.profile(entry_name, *input_args)

    out_nd = vm[entry_name](*input_args)

    # Warmup with 5 runs
    num_warms = 1
    for i in range(num_warms):
        vm[entry_name](*input_args)

    iterations = 3
    # measure perf
    start_time = time.time()
    for i in range(iterations):
        vm[entry_name](*input_args)
    duration = (time.time() - start_time) * 1000 / iterations

    def _to_numpy(outs):
        if isinstance(outs, tvm.nd.NDArray):
            return outs.asnumpy()
        ret = []
        if isinstance(outs, (list, tvm.ir.container.Array)):
            for out in outs:
                ret.append(_to_numpy(out))
        return ret

    out_np = _to_numpy(out_nd)

    print("inference result numpy[0][0]: ", out_np[0][0])
    print("Relax SAM inference output: ", out_np[0][0])
    print("Relax SAM inference performance: {} ms".format(duration))


def test_tvm_profile():
    batch_size, total_seq_len, dtype = 1, 32, "float32"

    mod_spec = {
        "get_image_embeddings": {
            # (batch_size, num_channels, height, width)
            "pixel_values": spec.Tensor([1, 3, 1024, 1024], "float32"),
        },
        "get_prompt_embeddings": {"input_points": spec.Tensor([1, 1, 1, 2], "float32")},
        "forward": {
            "pixel_values": spec.Tensor([1, 3, 1024, 1024], "float32"),
            "input_points": spec.Tensor([1, 1, 1, 2], "float32"),
        },
    }

    config = SamConfig()
    relax_model = RXSamModel(config)
    ir_mod, _ = relax_model.export_tvm(spec=mod_spec, debug=True)

    mod = tvm.ir.IRModule()
    mod["main"] = ir_mod["forward"]
    mod["get_prompt_embeddings"] = ir_mod["get_prompt_embeddings"]
    mod["get_image_embeddings"] = ir_mod["get_image_embeddings"]

    mod["_initialize_effect"] = ir_mod["_initialize_effect"]
    entry_name = "main"

    # target, dev = "llvm", tvm.cpu()
    target = tvm.target.Target("nvidia/nvidia-a10g")  # tvm.target.Target("cuda", host="llvm")
    dev = tvm.gpu()

    # apply passes
    mod = run_opt_passes(mod, combine_matmul=True)
    # mod = offload_to_cutlass(mod, target)

    # Apply cutlass optimization.
    # mod = partition_for_cutlass(mod)
    # mod = relax.transform.RunCodegen(
    #    {"cutlass": {"sm": 80, "find_first_valid": False}}
    # )(mod)

    mod = run_lower_passes(mod, target, do_tuning=False)

    # with tvm.target.Target("cuda"):
    #     mod = tvm.tir.transform.DefaultGPUSchedule()(mod)

    exe = relax.build(mod, target=target)
    # vm = relax.VirtualMachine(exe, dev)
    vm = relax.VirtualMachine(exe, dev, profile=True)

    # Prepare inputs for inference
    global pt_sam_model
    tvm_params = {}
    for k, v in pt_sam_model.state_dict().items():
        tvm_params[k] = tvm.nd.array(v.cpu().numpy(), dev)

    # image input
    img_inputs = get_inputs()
    for k, v in img_inputs.items():
        tvm_params[k] = tvm.nd.array(v.cpu().float().numpy(), dev)

    effects = vm["_initialize_effect"]()
    tvm_params[".io"] = effects

    # Convert param into ordered list.
    func_arity = vm._get_function_arity(entry_name)
    tvm_names = [vm._get_function_param_name(entry_name, i) for i in range(func_arity)]

    input_args = [tvm_params[name] for name in tvm_names]
    print("start profiling")

    # report = vm.profile(entry_name, *input_args)
    print("input_args[2] type : ", type(input_args[2]))
    print("effects type : ", type(effects))
    print("input_args[2]: ", input_args[2])
    print("effects: ", effects)
    vm.set_input(entry_name, *input_args)
    # print("set input works")
    report = vm.profile(entry_name)

    print("Profiling report: \n", report)


if __name__ == "__main__":
    # test_tvm_sam()
    test_tvm_profile()
