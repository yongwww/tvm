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
from tvm.relax.backend.contrib.cutlass import partition_for_cutlass

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


def get_transformers_torch_sam(type="base"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if type == "base":
        tmp_sam = PTSamModel.from_pretrained("facebook/sam-vit-base") #.to(device)
    else:
        tmp_sam = PTSamModel.from_pretrained("facebook/sam-vit-huge")
    
    config = tmp_sam.config
    return PTSamModel(config).to(device)
    # sam_huge_config = model.config


# device = "cuda" if torch.cuda.is_available() else "cpu"
# config = SamConfig()
# pt_sam_model = PTSamModel(config).to(device)


def test_transformers_sam(pt_sam_model=None):
    # print("input:  ", k, " shape:  ", v.shape) for k, v in inputs.data.items()
    inputs = get_inputs()

    # Warmup with 5 runs
    num_warms = 5
    for i in range(num_warms):
        pt_sam_model(**inputs)

    iterations = 15
    # measure perf
    start_time = time.time()
    for i in range(iterations):
        pt_sam_model(**inputs)
    duration = (time.time() - start_time) * 1000 / iterations

    outputs = pt_sam_model(**inputs)

    print("Hugging Face Transformers SAM inference output: ", outputs[0])
    print("Hugging Face Transformers SAM inference performance: {} ms".format(duration))


def test_transformers_sam_huge():
    # torch.backends.cuda.max_split_size_mb = 512
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PTSamModel.from_pretrained("facebook/sam-vit-huge").to(device)
    sam_huge_config = model.config
    print(sam_huge_config)
    relax_model = RXSamModel(sam_huge_config)
    """
    processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

    img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
    input_points = [[[450, 600]]]  # 2D location of a window in the image

    inputs = processor(raw_image, input_points=input_points, return_tensors="pt").to(device)
    # torch.cuda.empty_cache()
    outputs = model(**inputs)

    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )
    scores = outputs.iou_scores

    print("result score: ", scores)
    """


def _run_opt_passes(mod, params=None, fp16_input_names=None, combine_matmul=False):
    passes = [
        relax.transform.EliminateCommonSubexpr(),
        relax.transform.CanonicalizeBindings(),
        relax.transform.ConvertLayout({"relax.nn.conv2d": ["NHWC", "OHWI"]}),
        # get_rewrite_pass(combine_matmul),  # error
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


def test_tvm_sam(pt_sam_model=None):
    batch_size, total_seq_len, dtype = 1, 32, "float32"
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
    vm = relax.VirtualMachine(exe, dev)

    # Prepare inputs for inference
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
    num_warms = 5
    for i in range(num_warms):
        vm[entry_name](*input_args)

    iterations = 15
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

    pt_sam_model = get_transformers_torch_sam("huge")
    test_tvm_sam(pt_sam_model)
    test_transformers_sam(pt_sam_model)
    

    # test_tvm_sam()
    # test_transformers_sam()
    # test_transformers_sam_huge()
    # test_tvm_sam_huge()
    # test_tvm_profile()
