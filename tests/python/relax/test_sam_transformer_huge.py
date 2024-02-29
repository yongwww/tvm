import torch
from PIL import Image
import requests
import time

import tvm
from tvm import relax, tir
from tvm.relax.frontend.nn import spec
import numpy as np

import tvm.topi.testing
from tvm.relax.transform import LegalizeOps
from tvm.relax.frontend.nn.core import set_default_dtype as rx_set_default_dtype
from tvm.script import relax as R, tir as T
import tvm.testing
from tvm.relax.backend.contrib.cutlass import partition_for_cutlass

from transformers.models.sam.modeling_sam import SamModel as PTSamModel
from transformers.models.sam.modeling_tvm_sam import SamModel as RXSamModel
from transformers.models.sam.processing_sam import SamProcessor
from transformers.image_processing_utils import BatchFeature

from transformers.models.sam.configuration_sam import (
    SamConfig,
    SamVisionConfig,
    SamPromptEncoderConfig,
    SamMaskDecoderConfig,
)

from transformers.models.sam.clownfish_util import (
    run_opt_passes,
    offload_to_cutlass,
    run_lower_passes,
    partition_for_cutlass,
)


def _run_opt_passes(mod, params=None, fp16_input_names=None, combine_matmul=False):
    passes = [
        relax.transform.EliminateCommonSubexpr(),
        relax.transform.CanonicalizeBindings(),
        relax.transform.ConvertLayout({"relax.nn.conv2d": ["NHWC", "OHWI"]}),
        get_rewrite_pass(combine_matmul),  # error
        relax.transform.DeadCodeElimination(["main"]),
        #  File "/home/ubuntu/tvm/python/tvm/runtime/object.py", line 75, in __getattr__
        # raise AttributeError(f"{type(self)} has no attribute {name}") from None
        # AttributeError: <class 'tvm.relax.expr.DataflowVar'> has no attribute attrs
    ]
    """
         File "/home/ubuntu/tvm/src/relax/ir/transform.cc", line 285
        InternalError: Check failed: (global_scope_vars.empty() && symbolic_vars.empty()) is false:
          Error: DataflowBlock Pass should not delete any GlobalScope/Symbolic Var.
    """

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
            #    out_dtype="float16", fp16_input_names=fp16_input_names
            # ),
        ]
        """
              File "/home/ubuntu/tvm/src/relax/transform/infer_amp_utils.cc", line 28
              InternalError: Check failed: (tensor) is false: Expected TensorStructInfo, but got R.Objec
        """,

    return tvm.transform.Sequential(passes)(mod)


def _offload_to_cutlass(mod, target):
    # Currently, sm86 is not supported.
    sm = int(target.arch.split("_")[1])
    print("sm: ", sm)
    if sm > 80:
        sm = 80
    mod = partition_for_cutlass(mod)
    # print(mod.script(show_meta=True))
    mod = relax.transform.RunCodegen(
        {"cutlass": {"sm": sm, "find_first_valid": False}},
        entry_functions=["main", "get_prompt_embeddings", "get_image_embeddings"],
    )(mod)

    return mod


def get_inputs(dtype=torch.float32, size="base"):
    # dtype = torch.float16
    # size = "huge"
    torch.set_default_dtype(dtype)

    # Define a function to recursively convert a tensor to float16
    def _convert_dtype(value, dtype=torch.float32):
        if isinstance(value, torch.Tensor):
            if value.dtype == dtype:
                return value
            if (
                value.dtype == torch.float32
                or value.dtype == torch.float64
                or value.dtype == torch.float16
            ):
                return value.to(dtype)
            else:
                return value
        elif isinstance(value, dict):
            return {key: _convert_dtype(val, dtype) for key, val in value.items()}
        elif isinstance(value, list):
            return [_convert_dtype(item, dtype) for item in value]
        else:
            return value

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = SamProcessor.from_pretrained(
        "facebook/sam-vit-base" if size == "base" else "facebook/sam-vit-huge"
    )

    img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
    input_points = [[[450, 600]]]  # 2D location of a window in the image

    inputs = processor(
        raw_image, input_points=input_points, return_tensors="pt", torch_dtype=dtype
    ).to(device)

    inputs = BatchFeature({key: _convert_dtype(value, dtype) for key, value in inputs.items()})

    return inputs


def get_transformers_torch_sam(size="base", torch_dtype=torch.float32):
    # dtype = torch.float16
    torch.set_default_dtype(torch_dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if size == "base":
        tmp_sam = PTSamModel.from_pretrained(
            "facebook/sam-vit-base", torch_dtype=torch_dtype
        )  # .to(device)
    else:
        tmp_sam = PTSamModel.from_pretrained("facebook/sam-vit-huge", torch_dtype=torch_dtype)

    config = tmp_sam.config

    return PTSamModel(config).to(device)


def test_transformers_sam(pt_sam_model=None, torch_dtype=torch.float32, benchmark=True):
    inputs = get_inputs(torch_dtype)

    if benchmark is False:
        outputs = pt_sam_model(**inputs)
        print("Hugging Face Transformers SAM inference output: ", outputs)
        return outputs

    # Warmup with 5 runs
    num_warms = 5
    for i in range(num_warms):
        pt_sam_model(**inputs)
        # return

    iterations = 30
    # measure perf
    start_time = time.time()
    for i in range(iterations):
        pt_sam_model(**inputs)
    duration = (time.time() - start_time) * 1000 / iterations

    outputs = pt_sam_model(**inputs)

    print("Hugging Face Transformers SAM inference output: ", outputs[0])
    print("Hugging Face Transformers SAM inference performance: {} ms".format(duration))


def test_tvm_sam(pt_sam_model=None, dtype="float32", benchmark=True):
    rx_set_default_dtype(dtype)
    batch_size, total_seq_len, dtype = 1, 32, dtype
    # get the config
    config = pt_sam_model.config

    mod_spec = {
        "get_image_embeddings": {
            # (batch_size, num_channels, height, width)
            "pixel_values": spec.Tensor([1, 3, 1024, 1024], dtype),
        },
        "get_prompt_embeddings": {"input_points": spec.Tensor([1, 1, 1, 2], dtype)},
        "forward": {
            "pixel_values": spec.Tensor([1, 3, 1024, 1024], dtype),
            "input_points": spec.Tensor([1, 1, 1, 2], dtype),
        },
    }

    relax_model = RXSamModel(config)
    ir_mod, _ = relax_model.export_tvm(spec=mod_spec, debug=True)

    mod = tvm.ir.IRModule()
    mod["main"] = ir_mod["forward"].with_attrs({"global_symbol": "main"})
    mod["get_prompt_embeddings"] = ir_mod["get_prompt_embeddings"]
    mod["get_image_embeddings"] = ir_mod["get_image_embeddings"]

    mod["_initialize_effect"] = ir_mod["_initialize_effect"]
    entry_name = "main"

    # target, dev = "llvm", tvm.cpu()
    target = tvm.target.Target("nvidia/nvidia-a100")  # tvm.target.Target("cuda", host="llvm")
    dev = tvm.gpu()

    # apply passes
    mod = run_opt_passes(mod, combine_matmul=True)
    # print(mod.script(show_meta=True))

    # mod = _offload_to_cutlass(mod, target)

    # Apply cutlass optimization.
    # mod = partition_for_cutlass(mod)
    # mod = relax.transform.RunCodegen(
    #    {"cutlass": {"sm": 80, "find_first_valid": False}}
    # )(mod)

    mod = run_lower_passes(mod, target, do_tuning=False)

    # mod = relax.transform.LambdaLift()(mod)

    # with tvm.target.Target("cuda"):
    #     mod = tvm.tir.transform.DefaultGPUSchedule()(mod)

    # print(mod.script(show_meta=True))
    exe = relax.build(mod, target=target)
    vm = relax.VirtualMachine(exe, dev)

    # Prepare inputs for inference
    tvm_params = {}
    for k, v in pt_sam_model.state_dict().items():
        tvm_params[k] = tvm.nd.array(v.cpu().numpy(), dev)

    # image input
    torch_dtype = torch.float16 if dtype == "float16" else torch.float32
    img_inputs = get_inputs(torch_dtype)

    for k, v in img_inputs.items():
        tvm_params[k] = tvm.nd.array(v.cpu().numpy(), dev)

    effects = vm["_initialize_effect"]()
    tvm_params[".io"] = effects

    # Convert param into ordered list.
    func_arity = vm._get_function_arity(entry_name)
    tvm_names = [vm._get_function_param_name(entry_name, i) for i in range(func_arity)]

    input_args = [tvm_params[name] for name in tvm_names]
    # vm_profiler = relax.VirtualMachine(exe, tvm.gpu(), profile=True)
    # report = vm_profiler.profile(entry_name, *input_args)

    out_nd = vm[entry_name](*input_args)

    def _to_numpy(outs):
        if isinstance(outs, tvm.nd.NDArray):
            return outs.asnumpy()
        ret = []
        if isinstance(outs, (list, tvm.ir.container.Array)):
            for out in outs:
                ret.append(_to_numpy(out))
        return ret

    if benchmark:
        # Warmup with 5 runs
        num_warms = 5
        for i in range(num_warms):
            vm[entry_name](*input_args)
            # return

        iterations = 15
        # measure perf
        start_time = time.time()
        for i in range(iterations):
            vm[entry_name](*input_args)
        duration = (time.time() - start_time) * 1000 / iterations
        print("Relax SAM inference performance: {} ms".format(duration))

    out_np = _to_numpy(out_nd)

    # print("inference result numpy[0][0]: ", out_np[0][0])
    print("Relax SAM inference output: ", _to_numpy(out_nd))
    return out_nd[0]


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

    relax_model = RXSamModel(config)
    ir_mod, _ = relax_model.export_tvm(spec=mod_spec, debug=True)

    mod = tvm.ir.IRModule()
    mod["main"] = ir_mod["forward"]
    mod["get_prompt_embeddings"] = ir_mod["get_prompt_embeddings"]
    mod["get_image_embeddings"] = ir_mod["get_image_embeddings"]

    mod["_initialize_effect"] = ir_mod["_initialize_effect"]
    entry_name = "main"

    # target, dev = "llvm", tvm.cpu()
    target = tvm.target.Target("nvidia/nvidia-a100")  # tvm.target.Target("cuda", host="llvm")
    dev = tvm.gpu()

    # apply passes
    mod = run_opt_passes(mod, combine_matmul=True)
    # print(mod.script(show_meta=True))

    mod = _offload_to_cutlass(mod, target)

    # Apply cutlass optimization.
    # mod = partition_for_cutlass(mod)
    # mod = relax.transform.RunCodegen(
    #    {"cutlass": {"sm": 80, "find_first_valid": False}}
    # )(mod)

    mod = run_lower_passes(mod, target, do_tuning=False)

    # mod = relax.transform.LambdaLift()(mod)

    # with tvm.target.Target("cuda"):
    #     mod = tvm.tir.transform.DefaultGPUSchedule()(mod)

    # print(mod.script(show_meta=True))
    exe = relax.build(mod, target=target)
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

    vm.set_input(entry_name, *input_args)
    # print("set input works")
    report = vm.profile(entry_name)

    print("Profiling report: \n", report)


if __name__ == "__main__":
    # Test config
    dtype, size = "float16", "huge"  # "base"  # "huge"
    benchmark = True

    # Get the torch SAM
    torch_dtype = torch.float16 if dtype == "float16" else torch.float32
    pt_sam_model = get_transformers_torch_sam(size, torch_dtype=torch_dtype)
    from transformers.models.sam.modeling_sam import SamImageSegmentationOutput

    # run with PyTorch
    pt_out = test_transformers_sam(pt_sam_model, torch_dtype=torch_dtype, benchmark=benchmark)

    # convert to relax and run on relax vm
    tvm_out = test_tvm_sam(pt_sam_model, dtype=dtype, benchmark=benchmark)

    # Verify correctness
    tol = 1e-3
    if isinstance(tvm_out, tvm.container.Array):
        for i, (o1, o2) in enumerate(zip(tvm_out, pt_out)):
            print("Comparing output ", i)
            tvm.testing.assert_allclose(
                o1.numpy(), getattr(pt_out, o2, None).cpu().detach().numpy(), rtol=tol, atol=tol
            )
    else:
        tvm.testing.assert_allclose(tvm_out.numpy(), pt_out.cpu().numpy(), rtol=tol, atol=tol)

    print("Inference results match!")
