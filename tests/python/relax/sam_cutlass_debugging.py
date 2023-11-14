import torch
from PIL import Image
import requests
import time

import numpy as np

from transformers.models.sam.modeling_sam import SamModel as PTSamModel
from transformers.models.sam.processing_sam import SamProcessor
from transformers.models.sam.configuration_sam import SamConfig

import tvm
from tvm import relax, tir
from tvm.relax.frontend.nn import spec


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
    """
    passes = [
                relax.transform.LegalizeOps(),
                relax.transform.AnnotateTIROpPattern(),
                relax.transform.FoldConstant(),
                # relax.transform.FuseOps(),
                # relax.transform.FuseTIR(),
            ]
    """

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


metadata = tvm.ir.load_json("""{
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
      \"data\": [
        3, 
        14, 
        25, 
        36, 
        47, 
        58, 
        68, 
        79, 
        90, 
        101, 
        112
      ]
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
    }, 
    {
      \"type_key\": \"relax.expr.Constant\", 
      \"attrs\": {
        \"_checked_type_\": \"24\", 
        \"data\": \"1\", 
        \"span\": \"0\", 
        \"struct_info_\": \"15\"
      }
    }, 
    {
      \"type_key\": \"relax.TensorStructInfo\", 
      \"attrs\": {
        \"dtype\": \"bool\", 
        \"ndim\": \"4\", 
        \"shape\": \"16\", 
        \"span\": \"0\", 
        \"vdevice\": \"0\"
      }
    }, 
    {
      \"type_key\": \"relax.expr.ShapeExpr\", 
      \"attrs\": {
        \"_checked_type_\": \"23\", 
        \"span\": \"0\", 
        \"struct_info_\": \"22\", 
        \"values\": \"17\"
      }
    }, 
    {
      \"type_key\": \"Array\", 
      \"data\": [18, 19, 20, 21]
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
      \"type_key\": \"IntImm\", 
      \"attrs\": {
        \"dtype\": \"int64\", 
        \"span\": \"0\", 
        \"value\": \"1\"
      }
    }, 
    {
      \"type_key\": \"relax.ShapeStructInfo\", 
      \"attrs\": {
        \"ndim\": \"4\", 
        \"span\": \"0\", 
        \"values\": \"17\"
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
        \"dtype\": \"bool\", 
        \"ndim\": \"4\", 
        \"span\": \"0\"
      }
    }, 
    {
      \"type_key\": \"relax.expr.Constant\", 
      \"attrs\": {
        \"_checked_type_\": \"35\", 
        \"data\": \"2\", 
        \"span\": \"0\", 
        \"struct_info_\": \"26\"
      }
    }, 
    {
      \"type_key\": \"relax.TensorStructInfo\", 
      \"attrs\": {
        \"dtype\": \"bool\", 
        \"ndim\": \"4\", 
        \"shape\": \"27\", 
        \"span\": \"0\", 
        \"vdevice\": \"0\"
      }
    }, 
    {
      \"type_key\": \"relax.expr.ShapeExpr\", 
      \"attrs\": {
        \"_checked_type_\": \"34\", 
        \"span\": \"0\", 
        \"struct_info_\": \"33\", 
        \"values\": \"28\"
      }
    }, 
    {
      \"type_key\": \"Array\", 
      \"data\": [29, 30, 31, 32]
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
      \"type_key\": \"IntImm\", 
      \"attrs\": {
        \"dtype\": \"int64\", 
        \"span\": \"0\", 
        \"value\": \"1\"
      }
    }, 
    {
      \"type_key\": \"relax.ShapeStructInfo\", 
      \"attrs\": {
        \"ndim\": \"4\", 
        \"span\": \"0\", 
        \"values\": \"28\"
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
        \"dtype\": \"bool\", 
        \"ndim\": \"4\", 
        \"span\": \"0\"
      }
    }, 
    {
      \"type_key\": \"relax.expr.Constant\", 
      \"attrs\": {
        \"_checked_type_\": \"46\", 
        \"data\": \"3\", 
        \"span\": \"0\", 
        \"struct_info_\": \"37\"
      }
    }, 
    {
      \"type_key\": \"relax.TensorStructInfo\", 
      \"attrs\": {
        \"dtype\": \"bool\", 
        \"ndim\": \"4\", 
        \"shape\": \"38\", 
        \"span\": \"0\", 
        \"vdevice\": \"0\"
      }
    }, 
    {
      \"type_key\": \"relax.expr.ShapeExpr\", 
      \"attrs\": {
        \"_checked_type_\": \"45\", 
        \"span\": \"0\", 
        \"struct_info_\": \"44\", 
        \"values\": \"39\"
      }
    }, 
    {
      \"type_key\": \"Array\", 
      \"data\": [40, 41, 42, 43]
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
      \"type_key\": \"IntImm\", 
      \"attrs\": {
        \"dtype\": \"int64\", 
        \"span\": \"0\", 
        \"value\": \"1\"
      }
    }, 
    {
      \"type_key\": \"relax.ShapeStructInfo\", 
      \"attrs\": {
        \"ndim\": \"4\", 
        \"span\": \"0\", 
        \"values\": \"39\"
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
        \"dtype\": \"bool\", 
        \"ndim\": \"4\", 
        \"span\": \"0\"
      }
    }, 
    {
      \"type_key\": \"relax.expr.Constant\", 
      \"attrs\": {
        \"_checked_type_\": \"57\", 
        \"data\": \"4\", 
        \"span\": \"0\", 
        \"struct_info_\": \"48\"
      }
    }, 
    {
      \"type_key\": \"relax.TensorStructInfo\", 
      \"attrs\": {
        \"dtype\": \"bool\", 
        \"ndim\": \"4\", 
        \"shape\": \"49\", 
        \"span\": \"0\", 
        \"vdevice\": \"0\"
      }
    }, 
    {
      \"type_key\": \"relax.expr.ShapeExpr\", 
      \"attrs\": {
        \"_checked_type_\": \"56\", 
        \"span\": \"0\", 
        \"struct_info_\": \"55\", 
        \"values\": \"50\"
      }
    }, 
    {
      \"type_key\": \"Array\", 
      \"data\": [51, 52, 53, 54]
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
      \"type_key\": \"IntImm\", 
      \"attrs\": {
        \"dtype\": \"int64\", 
        \"span\": \"0\", 
        \"value\": \"1\"
      }
    }, 
    {
      \"type_key\": \"relax.ShapeStructInfo\", 
      \"attrs\": {
        \"ndim\": \"4\", 
        \"span\": \"0\", 
        \"values\": \"50\"
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
        \"dtype\": \"bool\", 
        \"ndim\": \"4\", 
        \"span\": \"0\"
      }
    }, 
    {
      \"type_key\": \"relax.expr.Constant\", 
      \"attrs\": {
        \"_checked_type_\": \"67\", 
        \"data\": \"5\", 
        \"span\": \"0\", 
        \"struct_info_\": \"59\"
      }
    }, 
    {
      \"type_key\": \"relax.TensorStructInfo\", 
      \"attrs\": {
        \"dtype\": \"float32\", 
        \"ndim\": \"3\", 
        \"shape\": \"60\", 
        \"span\": \"0\", 
        \"vdevice\": \"0\"
      }
    }, 
    {
      \"type_key\": \"relax.expr.ShapeExpr\", 
      \"attrs\": {
        \"_checked_type_\": \"66\", 
        \"span\": \"0\", 
        \"struct_info_\": \"65\", 
        \"values\": \"61\"
      }
    }, 
    {
      \"type_key\": \"Array\", 
      \"data\": [62, 63, 64]
    }, 
    {
      \"type_key\": \"IntImm\", 
      \"attrs\": {
        \"dtype\": \"int64\", 
        \"span\": \"0\", 
        \"value\": \"64\"
      }
    }, 
    {
      \"type_key\": \"IntImm\", 
      \"attrs\": {
        \"dtype\": \"int64\", 
        \"span\": \"0\", 
        \"value\": \"64\"
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
        \"ndim\": \"3\", 
        \"span\": \"0\", 
        \"values\": \"61\"
      }
    }, 
    {
      \"type_key\": \"relax.ShapeType\", 
      \"attrs\": {
        \"ndim\": \"3\", 
        \"span\": \"0\"
      }
    }, 
    {
      \"type_key\": \"relax.DynTensorType\", 
      \"attrs\": {
        \"dtype\": \"float32\", 
        \"ndim\": \"3\", 
        \"span\": \"0\"
      }
    }, 
    {
      \"type_key\": \"relax.expr.Constant\", 
      \"attrs\": {
        \"_checked_type_\": \"78\", 
        \"data\": \"6\", 
        \"span\": \"0\", 
        \"struct_info_\": \"69\"
      }
    }, 
    {
      \"type_key\": \"relax.TensorStructInfo\", 
      \"attrs\": {
        \"dtype\": \"float32\", 
        \"ndim\": \"4\", 
        \"shape\": \"70\", 
        \"span\": \"0\", 
        \"vdevice\": \"0\"
      }
    }, 
    {
      \"type_key\": \"relax.expr.ShapeExpr\", 
      \"attrs\": {
        \"_checked_type_\": \"77\", 
        \"span\": \"0\", 
        \"struct_info_\": \"76\", 
        \"values\": \"71\"
      }
    }, 
    {
      \"type_key\": \"Array\", 
      \"data\": [72, 73, 74, 75]
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
        \"values\": \"71\"
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
    }, 
    {
      \"type_key\": \"relax.expr.Constant\", 
      \"attrs\": {
        \"_checked_type_\": \"89\", 
        \"data\": \"7\", 
        \"span\": \"0\", 
        \"struct_info_\": \"80\"
      }
    }, 
    {
      \"type_key\": \"relax.TensorStructInfo\", 
      \"attrs\": {
        \"dtype\": \"bool\", 
        \"ndim\": \"4\", 
        \"shape\": \"81\", 
        \"span\": \"0\", 
        \"vdevice\": \"0\"
      }
    }, 
    {
      \"type_key\": \"relax.expr.ShapeExpr\", 
      \"attrs\": {
        \"_checked_type_\": \"88\", 
        \"span\": \"0\", 
        \"struct_info_\": \"87\", 
        \"values\": \"82\"
      }
    }, 
    {
      \"type_key\": \"Array\", 
      \"data\": [83, 84, 85, 86]
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
      \"type_key\": \"IntImm\", 
      \"attrs\": {
        \"dtype\": \"int64\", 
        \"span\": \"0\", 
        \"value\": \"1\"
      }
    }, 
    {
      \"type_key\": \"relax.ShapeStructInfo\", 
      \"attrs\": {
        \"ndim\": \"4\", 
        \"span\": \"0\", 
        \"values\": \"82\"
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
        \"dtype\": \"bool\", 
        \"ndim\": \"4\", 
        \"span\": \"0\"
      }
    }, 
    {
      \"type_key\": \"relax.expr.Constant\", 
      \"attrs\": {
        \"_checked_type_\": \"100\", 
        \"data\": \"8\", 
        \"span\": \"0\", 
        \"struct_info_\": \"91\"
      }
    }, 
    {
      \"type_key\": \"relax.TensorStructInfo\", 
      \"attrs\": {
        \"dtype\": \"bool\", 
        \"ndim\": \"4\", 
        \"shape\": \"92\", 
        \"span\": \"0\", 
        \"vdevice\": \"0\"
      }
    }, 
    {
      \"type_key\": \"relax.expr.ShapeExpr\", 
      \"attrs\": {
        \"_checked_type_\": \"99\", 
        \"span\": \"0\", 
        \"struct_info_\": \"98\", 
        \"values\": \"93\"
      }
    }, 
    {
      \"type_key\": \"Array\", 
      \"data\": [94, 95, 96, 97]
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
      \"type_key\": \"IntImm\", 
      \"attrs\": {
        \"dtype\": \"int64\", 
        \"span\": \"0\", 
        \"value\": \"1\"
      }
    }, 
    {
      \"type_key\": \"relax.ShapeStructInfo\", 
      \"attrs\": {
        \"ndim\": \"4\", 
        \"span\": \"0\", 
        \"values\": \"93\"
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
        \"dtype\": \"bool\", 
        \"ndim\": \"4\", 
        \"span\": \"0\"
      }
    }, 
    {
      \"type_key\": \"relax.expr.Constant\", 
      \"attrs\": {
        \"_checked_type_\": \"111\", 
        \"data\": \"9\", 
        \"span\": \"0\", 
        \"struct_info_\": \"102\"
      }
    }, 
    {
      \"type_key\": \"relax.TensorStructInfo\", 
      \"attrs\": {
        \"dtype\": \"bool\", 
        \"ndim\": \"4\", 
        \"shape\": \"103\", 
        \"span\": \"0\", 
        \"vdevice\": \"0\"
      }
    }, 
    {
      \"type_key\": \"relax.expr.ShapeExpr\", 
      \"attrs\": {
        \"_checked_type_\": \"110\", 
        \"span\": \"0\", 
        \"struct_info_\": \"109\", 
        \"values\": \"104\"
      }
    }, 
    {
      \"type_key\": \"Array\", 
      \"data\": [105, 106, 107, 108]
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
      \"type_key\": \"IntImm\", 
      \"attrs\": {
        \"dtype\": \"int64\", 
        \"span\": \"0\", 
        \"value\": \"1\"
      }
    }, 
    {
      \"type_key\": \"relax.ShapeStructInfo\", 
      \"attrs\": {
        \"ndim\": \"4\", 
        \"span\": \"0\", 
        \"values\": \"104\"
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
        \"dtype\": \"bool\", 
        \"ndim\": \"4\", 
        \"span\": \"0\"
      }
    }, 
    {
      \"type_key\": \"relax.expr.Constant\", 
      \"attrs\": {
        \"_checked_type_\": \"122\", 
        \"data\": \"10\", 
        \"span\": \"0\", 
        \"struct_info_\": \"113\"
      }
    }, 
    {
      \"type_key\": \"relax.TensorStructInfo\", 
      \"attrs\": {
        \"dtype\": \"bool\", 
        \"ndim\": \"4\", 
        \"shape\": \"114\", 
        \"span\": \"0\", 
        \"vdevice\": \"0\"
      }
    }, 
    {
      \"type_key\": \"relax.expr.ShapeExpr\", 
      \"attrs\": {
        \"_checked_type_\": \"121\", 
        \"span\": \"0\", 
        \"struct_info_\": \"120\", 
        \"values\": \"115\"
      }
    }, 
    {
      \"type_key\": \"Array\", 
      \"data\": [116, 117, 118, 119]
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
      \"type_key\": \"IntImm\", 
      \"attrs\": {
        \"dtype\": \"int64\", 
        \"span\": \"0\", 
        \"value\": \"1\"
      }
    }, 
    {
      \"type_key\": \"relax.ShapeStructInfo\", 
      \"attrs\": {
        \"ndim\": \"4\", 
        \"span\": \"0\", 
        \"values\": \"115\"
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
        \"dtype\": \"bool\", 
        \"ndim\": \"4\", 
        \"span\": \"0\"
      }
    }
  ], 
  \"b64ndarrays\": [
    \"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AABUPwAATL8AAFw/AABMvwAAZD8AAEy/AABsPwAATL8AAHQ/AABMvwAAfD8AAEy/AAB8vwAARL8AAHS/AABEvwAAbL8AAES/AABkvwAARL8AAFy/AABEvwAAVL8AAES/AABMvwAARL8AAES/AABEvwAAPL8AAES/AAA0vwAARL8AACy/AABEvwAAJL8AAES/AAAcvwAARL8AABS/AABEvwAADL8AAES/AAAEvwAARL8AAPi+AABEvwAA6L4AAES/AADYvgAARL8AAMi+AABEvwAAuL4AAES/AACovgAARL8AAJi+AABEvwAAiL4AAES/AABwvgAARL8AAFC+AABEvwAAML4AAES/AAAQvgAARL8AAOC9AABEvwAAoL0AAES/AABAvQAARL8AAIC8AABEvwAAgDwAAES/AABAPQAARL8AAKA9AABEvwAA4D0AAES/AAAQPgAARL8AADA+AABEvwAAUD4AAES/AABwPgAARL8AAIg+AABEvwAAmD4AAES/AACoPgAARL8AALg+AABEvwAAyD4AAES/AADYPgAARL8AAOg+AABEvwAA+D4AAES/AAAEPwAARL8AAAw/AABEvwAAFD8AAES/AAAcPwAARL8AACQ/AABEvwAALD8AAES/AAA0PwAARL8AADw/AABEvwAARD8AAES/AABMPwAARL8AAFQ/AABEvwAAXD8AAES/AABkPwAARL8AAGw/AABEvwAAdD8AAES/AAB8PwAARL8AAHy/AAA8vwAAdL8AADy/AABsvwAAPL8AAGS/AAA8vwAAXL8AADy/AABUvwAAPL8AAEy/AAA8vwAARL8AADy/AAA8vwAAPL8AADS/AAA8vwAALL8AADy/AAAkvwAAPL8AABy/AAA8vwAAFL8AADy/AAAMvwAAPL8AAAS/AAA8vwAA+L4AADy/AADovgAAPL8AANi+AAA8vwAAyL4AADy/AAC4vgAAPL8AAKi+AAA8vwAAmL4AADy/AACIvgAAPL8AAHC+AAA8vwAAUL4AADy/AAAwvgAAPL8AABC+AAA8vwAA4L0AADy/AACgvQAAPL8AAEC9AAA8vwAAgLwAADy/AACAPAAAPL8AAEA9AAA8vwAAoD0AADy/AADgPQAAPL8AABA+AAA8vwAAMD4AADy/AABQPgAAPL8AAHA+AAA8vwAAiD4AADy/AACYPgAAPL8AAKg+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AACgvQAA6L4AAEC9AADovgAAgLwAAOi+AACAPAAA6L4AAEA9AADovgAAoD0AAOi+AADgPQAA6L4AABA+AADovgAAMD4AAOi+AABQPgAA6L4AAHA+AADovgAAiD4AAOi+AACYPgAA6L4AAKg+AADovgAAuD4AAOi+AADIPgAA6L4AANg+AADovgAA6D4AAOi+AAD4PgAA6L4AAAQ/AADovgAADD8AAOi+AAAUPwAA6L4AABw/AADovgAAJD8AAOi+AAAsPwAA6L4AADQ/AADovgAAPD8AAOi+AABEPwAA6L4AAEw/AADovgAAVD8AAOi+AABcPwAA6L4AAGQ/AADovgAAbD8AAOi+AAB0PwAA6L4AAHw/AADovgAAfL8AANi+AAB0vwAA2L4AAGy/AADYvgAAZL8AANi+AABcvwAA2L4AAFS/AADYvgAATL8AANi+AABEvwAA2L4AADy/AADYvgAANL8AANi+AAAsvwAA2L4AACS/AADYvgAAHL8AANi+AAAUvwAA2L4AAAy/AADYvgAABL8AANi+AAD4vgAA2L4AAOi+AADYvgAA2L4AANi+AADIvgAA2L4AALi+AADYvgAAqL4AANi+AACYvgAA2L4AAIi+AADYvgAAcL4AANi+AABQvgAA2L4AADC+AADYvgAAEL4AANi+AADgvQAA2L4AAKC9AADYvgAAQL0AANi+AACAvAAA2L4AAIA8AADYvgAAQD0AANi+AACgPQAA2L4AAOA9AADYvgAAED4AANi+AAAwPgAA2L4AAFA+AADYvgAAcD4AANi+AACIPgAA2L4AAJg+AADYvgAAqD4AANi+AAC4PgAA2L4AAMg+AADYvgAA2D4AANi+AADoPgAA2L4AAPg+AADYvgAABD8AANi+AAAMPwAA2L4AABQ/AADYvgAAHD8AANi+AAAkPwAA2L4AACw/AADYvgAAND8AANi+AAA8PwAA2L4AAEQ/AADYvgAATD8AANi+AABUPwAA2L4AAFw/AADYvgAAZD8AANi+AABsPwAA2L4AAHQ/AADYvgAAfD8AANi+AAB8vwAAyL4AAHS/AADIvgAAbL8AAMi+AABkvwAAyL4AAFy/AADIvgAAVL8AAMi+AABMvwAAyL4AAES/AADIvgAAPL8AAMi+AAA0vwAAyL4AACy/AADIvgAAJL8AAMi+AAAcvwAAyL4AABS/AADIvgAADL8AAMi+AAAEvwAAyL4AAPi+AADIvgAA6L4AAMi+AADYvgAAyL4AAMi+AADIvgAAuL4AAMi+AACovgAAyL4AAJi+AADIvgAAiL4AAMi+AABwvgAAyL4AAFC+AADIvgAAML4AAMi+AAAQvgAAyL4AAOC9AADIvgAAoL0AAMi+AABAvQAAyL4AAIC8AADIvgAAgDwAAMi+AABAPQAAyL4AAKA9AADIvgAA4D0AAMi+AAAQPgAAyL4AADA+AADIvgAAUD4AAMi+AABwPgAAyL4AAIg+AADIvgAAmD4AAMi+AACoPgAAyL4AALg+AADIvgAAyD4AAMi+AADYPgAAyL4AAOg+AADIvgAA+D4AAMi+AAAEPwAAyL4AAAw/AADIvgAAFD8AAMi+AAAcPwAAyL4AACQ/AADIvgAALD8AAMi+AAA0PwAAyL4AADw/AADIvgAARD8AAMi+AABMPwAAyL4AAFQ/AADIvgAAXD8AAMi+AABkPwAAyL4AAGw/AADIvgAAdD8AAMi+AAB8PwAAyL4AAHy/AAC4vgAAdL8AALi+AABsvwAAuL4AAGS/AAC4vgAAXL8AALi+AABUvwAAuL4AAEy/AAC4vgAARL8AALi+AAA8vwAAuL4AADS/AAC4vgAALL8AALi+AAAkvwAAuL4AABy/AAC4vgAAFL8AALi+AAAMvwAAuL4AAAS/AAC4vgAA+L4AALi+AADovgAAuL4AANi+AAC4vgAAyL4AALi+AAC4vgAAuL4AAKi+AAC4vgAAmL4AALi+AACIvgAAuL4AAHC+AAC4vgAAUL4AALi+AAAwvgAAuL4AABC+AAC4vgAA4L0AALi+AACgvQAAuL4AAEC9AAC4vgAAgLwAALi+AACAPAAAuL4AAEA9AAC4vgAAoD0AALi+AADgPQAAuL4AABA+AAC4vgAAMD4AALi+AABQPgAAuL4AAHA+AAC4vgAAiD4AALi+AACYPgAAuL4AAKg+AAC4vgAAuD4AALi+AADIPgAAuL4AANg+AAC4vgAA6D4AALi+AAD4PgAAuL4AAAQ/AAC4vgAADD8AALi+AAAUPwAAuL4AABw/AAC4vgAAJD8AALi+AAAsPwAAuL4AADQ/AAC4vgAAPD8AALi+AABEPwAAuL4AAEw/AAC4vgAAVD8AALi+AABcPwAAuL4AAGQ/AAC4vgAAbD8AALi+AAB0PwAAuL4AAHw/AAC4vgAAfL8AAKi+AAB0vwAAqL4AAGy/AACovgAAZL8AAKi+AABcvwAAqL4AAFS/AACovgAATL8AAKi+AABEvwAAqL4AADy/AACovgAANL8AAKi+AAAsvwAAqL4AACS/AACovgAAHL8AAKi+AAAUvwAAqL4AAAy/AACovgAABL8AAKi+AAD4vgAAqL4AAOi+AACovgAA2L4AAKi+AADIvgAAqL4AALi+AACovgAAqL4AAKi+AACYvgAAqL4AAIi+AACovgAAcL4AAKi+AABQvgAAqL4AADC+AACovgAAEL4AAKi+AADgvQAAqL4AAKC9AACovgAAQL0AAKi+AACAvAAAqL4AAIA8AACovgAAQD0AAKi+AACgPQAAqL4AAOA9AACovgAAED4AAKi+AAAwPgAAqL4AAFA+AACovgAAcD4AAKi+AACIPgAAqL4AAJg+AACovgAAqD4AAKi+AAC4PgAAqL4AAMg+AACovgAA2D4AAKi+AADoPgAAqL4AAPg+AACovgAABD8AAKi+AAAMPwAAqL4AABQ/AACovgAAHD8AAKi+AAAkPwAAqL4AACw/AACovgAAND8AAKi+AAA8PwAAqL4AAEQ/AACovgAATD8AAKi+AABUPwAAqL4AAFw/AACovgAAZD8AAKi+AABsPwAAqL4AAHQ/AACovgAAfD8AAKi+AAB8vwAAmL4AAHS/AACYvgAAbL8AAJi+AABkvwAAmL4AAFy/AACYvgAAVL8AAJi+AABMvwAAmL4AAES/AACYvgAAPL8AAJi+AAA0vwAAmL4AACy/AACYvgAAJL8AAJi+AAAcvwAAmL4AABS/AACYvgAADL8AAJi+AAAEvwAAmL4AAPi+AACYvgAA6L4AAJi+AADYvgAAmL4AAMi+AACYvgAAuL4AAJi+AACovgAAmL4AAJi+AACYvgAAiL4AAJi+AABwvgAAmL4AAFC+AACYvgAAML4AAJi+AAAQvgAAmL4AAOC9AACYvgAAoL0AAJi+AABAvQAAmL4AAIC8AACYvgAAgDwAAJi+AABAPQAAmL4AAKA9AACYvgAA4D0AAJi+AAAQPgAAmL4AADA+AACYvgAAUD4AAJi+AABwPgAAmL4AAIg+AACYvgAAmD4AAJi+AACoPgAAmL4AALg+AACYvgAAyD4AAJi+AADYPgAAmL4AAOg+AACYvgAA+D4AAJi+AAAEPwAAmL4AAAw/AACYvgAAFD8AAJi+AAAcPwAAmL4AACQ/AACYvgAALD8AAJi+AAA0PwAAmL4AADw/AACYvgAARD8AAJi+AABMPwAAmL4AAFQ/AACYvgAAXD8AAJi+AABkPwAAmL4AAGw/AACYvgAAdD8AAJi+AAB8PwAAmL4AAHy/AACIvgAAdL8AAIi+AABsvwAAiL4AAGS/AACIvgAAXL8AAIi+AABUvwAAiL4AAEy/AACIvgAARL8AAIi+AAA8vwAAiL4AADS/AACIvgAALL8AAIi+AAAkvwAAiL4AABy/AACIvgAAFL8AAIi+AAAMvwAAiL4AAAS/AACIvgAA+L4AAIi+AADovgAAiL4AANi+AACIvgAAyL4AAIi+AAC4vgAAiL4AAKi+AACIvgAAmL4AAIi+AACIvgAAiL4AAHC+AACIvgAAUL4AAIi+AAAwvgAAiL4AABC+AACIvgAA4L0AAIi+AACgvQAAiL4AAEC9AACIvgAAgLwAAIi+AACAPAAAiL4AAEA9AACIvgAAoD0AAIi+AADgPQAAiL4AABA+AACIvgAAMD4AAIi+AABQPgAAiL4AAHA+AACIvgAAiD4AAIi+AACYPgAAiL4AAKg+AACIvgAAuD4AAIi+AADIPgAAiL4AANg+AACIvgAA6D4AAIi+AAD4PgAAiL4AAAQ/AACIvgAADD8AAIi+AAAUPwAAiL4AABw/AACIvgAAJD8AAIi+AAAsPwAAiL4AADQ/AACIvgAAPD8AAIi+AABEPwAAiL4AAEw/AACIvgAAVD8AAIi+AABcPwAAiL4AAGQ/AACIvgAAbD8AAIi+AAB0PwAAiL4AAHw/AACIvgAAfL8AAHC+AAB0vwAAcL4AAGy/AABwvgAAZL8AAHC+AABcvwAAcL4AAFS/AABwvgAATL8AAHC+AABEvwAAcL4AADy/AABwvgAANL8AAHC+AAAsvwAAcL4AACS/AABwvgAAHL8AAHC+AAAUvwAAcL4AAAy/AABwvgAABL8AAHC+AAD4vgAAcL4AAOi+AABwvgAA2L4AAHC+AADIvgAAcL4AALi+AABwvgAAqL4AAHC+AACYvgAAcL4AAIi+AABwvgAAcL4AAHC+AABQvgAAcL4AADC+AABwvgAAEL4AAHC+AADgvQAAcL4AAKC9AABwvgAAQL0AAHC+AACAvAAAcL4AAIA8AABwvgAAQD0AAHC+AACgPQAAcL4AAOA9AABwvgAAED4AAHC+AAAwPgAAcL4AAFA+AABwvgAAcD4AAHC+AACIPgAAcL4AAJg+AABwvgAAqD4AAHC+AAC4PgAAcL4AAMg+AABwvgAA2D4AAHC+AADoPgAAcL4AAPg+AABwvgAABD8AAHC+AAAMPwAAcL4AABQ/AABwvgAAHD8AAHC+AAAkPwAAcL4AACw/AABwvgAAND8AAHC+AAA8PwAAcL4AAEQ/AABwvgAATD8AAHC+AABUPwAAcL4AAFw/AABwvgAAZD8AAHC+AABsPwAAcL4AAHQ/AABwvgAAfD8AAHC+AAB8vwAAUL4AAHS/AABQvgAAbL8AAFC+AABkvwAAUL4AAFy/AABQvgAAVL8AAFC+AABMvwAAUL4AAES/AABQvgAAPL8AAFC+AAA0vwAAUL4AACy/AABQvgAAJL8AAFC+AAAcvwAAUL4AABS/AABQvgAADL8AAFC+AAAEvwAAUL4AAPi+AABQvgAA6L4AAFC+AADYvgAAUL4AAMi+AABQvgAAuL4AAFC+AACovgAAUL4AAJi+AABQvgAAiL4AAFC+AABwvgAAUL4AAFC+AABQvgAAML4AAFC+AAAQvgAAUL4AAOC9AABQvgAAoL0AAFC+AABAvQAAUL4AAIC8AABQvgAAgDwAAFC+AABAPQAAUL4AAKA9AABQvgAA4D0AAFC+AAAQPgAAUL4AADA+AABQvgAAUD4AAFC+AABwPgAAUL4AAIg+AABQvgAAmD4AAFC+AACoPgAAUL4AALg+AABQvgAAyD4AAFC+AADYPgAAUL4AAOg+AABQvgAA+D4AAFC+AAAEPwAAUL4AAAw/AABQvgAAFD8AAFC+AAAcPwAAUL4AACQ/AABQvgAALD8AAFC+AAA0PwAAUL4AADw/AABQvgAARD8AAFC+AABMPwAAUL4AAFQ/AABQvgAAXD8AAFC+AABkPwAAUL4AAGw/AABQvgAAdD8AAFC+AAB8PwAAUL4AAHy/AAAwvgAAdL8AADC+AABsvwAAML4AAGS/AAAwvgAAXL8AADC+AABUvwAAML4AAEy/AAAwvgAARL8AADC+AAA8vwAAML4AADS/AAAwvgAALL8AADC+AAAkvwAAML4AABy/AAAwvgAAFL8AADC+AAAMvwAAML4AAAS/AAAwvgAA+L4AADC+AADovgAAML4AANi+AAAwvgAAyL4AADC+AAC4vgAAML4AAKi+AAAwvgAAmL4AADC+AACIvgAAML4AAHC+AAAwvgAAUL4AADC+AAAwvgAAML4AABC+AAAwvgAA4L0AADC+AACgvQAAML4AAEC9AAAwvgAAgLwAADC+AACAPAAAML4AAEA9AAAwvgAAoD0AADC+AADgPQAAML4AABA+AAAwvgAAMD4AADC+AABQPgAAML4AAHA+AAAwvgAAiD4AADC+AACYPgAAML4AAKg+AAAwvgAAuD4AADC+AADIPgAAML4AANg+AAAwvgAA6D4AADC+AAD4PgAAML4AAAQ/AAAwvgAADD8AADC+AAAUPwAAML4AABw/AAAwvgAAJD8AADC+AAAsPwAAML4AADQ/AAAwvgAAPD8AADC+AABEPwAAML4AAEw/AAAwvgAAVD8AADC+AABcPwAAML4AAGQ/AAAwvgAAbD8AADC+AAB0PwAAML4AAHw/AAAwvgAAfL8AABC+AAB0vwAAEL4AAGy/AAAQvgAAZL8AABC+AABcvwAAEL4AAFS/AAAQvgAATL8AABC+AABEvwAAEL4AADy/AAAQvgAANL8AABC+AAAsvwAAEL4AACS/AAAQvgAAHL8AABC+AAAUvwAAEL4AAAy/AAAQvgAABL8AABC+AAD4vgAAEL4AAOi+AAAQvgAA2L4AABC+AADIvgAAEL4AALi+AAAQvgAAqL4AABC+AACYvgAAEL4AAIi+AAAQvgAAcL4AABC+AABQvgAAEL4AADC+AAAQvgAAEL4AABC+AADgvQAAEL4AAKC9AAAQvgAAQL0AABC+AACAvAAAEL4AAIA8AAAQvgAAQD0AABC+AACgPQAAEL4AAOA9AAAQvgAAED4AABC+AAAwPgAAEL4AAFA+AAAQvgAAcD4AABC+AACIPgAAEL4AAJg+AAAQvgAAqD4AABC+AAC4PgAAEL4AAMg+AAAQvgAA2D4AABC+AADoPgAAEL4AAPg+AAAQvgAABD8AABC+AAAMPwAAEL4AABQ/AAAQvgAAHD8AABC+AAAkPwAAEL4AACw/AAAQvgAAND8AABC+AAA8PwAAEL4AAEQ/AAAQvgAATD8AABC+AABUPwAAEL4AAFw/AAAQvgAAZD8AABC+AABsPwAAEL4AAHQ/AAAQvgAAfD8AABC+AAB8vwAA4L0AAHS/AADgvQAAbL8AAOC9AABkvwAA4L0AAFy/AADgvQAAVL8AAOC9AABMvwAA4L0AAES/AADgvQAAPL8AAOC9AAA0vwAA4L0AACy/AADgvQAAJL8AAOC9AAAcvwAA4L0AABS/AADgvQAADL8AAOC9AAAEvwAA4L0AAPi+AADgvQAA6L4AAOC9AADYvgAA4L0AAMi+AADgvQAAuL4AAOC9AACovgAA4L0AAJi+AADgvQAAiL4AAOC9AABwvgAA4L0AAFC+AADgvQAAML4AAOC9AAAQvgAA4L0AAOC9AADgvQAAoL0AAOC9AABAvQAA4L0AAIC8AADgvQAAgDwAAOC9AABAPQAA4L0AAKA9AADgvQAA4D0AAOC9AAAQPgAA4L0AADA+AADgvQAAUD4AAOC9AABwPgAA4L0AAIg+AADgvQAAmD4AAOC9AACoPgAA4L0AALg+AADgvQAAyD4AAOC9AADYPgAA4L0AAOg+AADgvQAA+D4AAOC9AAAEPwAA4L0AAAw/AADgvQAAFD8AAOC9AAAcPwAA4L0AACQ/AADgvQAALD8AAOC9AAA0PwAA4L0AADw/AADgvQAARD8AAOC9AABMPwAA4L0AAFQ/AADgvQAAXD8AAOC9AABkPwAA4L0AAGw/AADgvQAAdD8AAOC9AAB8PwAA4L0AAHy/AACgvQAAdL8AAKC9AABsvwAAoL0AAGS/AACgvQAAXL8AAKC9AABUvwAAoL0AAEy/AACgvQAARL8AAKC9AAA8vwAAoL0AADS/AACgvQAALL8AAKC9AAAkvwAAoL0AABy/AACgvQAAFL8AAKC9AAAMvwAAoL0AAAS/AACgvQAA+L4AAKC9AADovgAAoL0AANi+AACgvQAAyL4AAKC9AAC4vgAAoL0AAKi+AACgvQAAmL4AAKC9AACIvgAAoL0AAHC+AACgvQAAUL4AAKC9AAAwvgAAoL0AABC+AACgvQAA4L0AAKC9AACgvQAAoL0AAEC9AACgvQAAgLwAAKC9AACAPAAAoL0AAEA9AACgvQAAoD0AAKC9AADgPQAAoL0AABA+AACgvQAAMD4AAKC9AABQPgAAoL0AAHA+AACgvQAAiD4AAKC9AACYPgAAoL0AAKg+AACgvQAAuD4AAKC9AADIPgAAoL0AANg+AACgvQAA6D4AAKC9AAD4PgAAoL0AAAQ/AACgvQAADD8AAKC9AAAUPwAAoL0AABw/AACgvQAAJD8AAKC9AAAsPwAAoL0AADQ/AACgvQAAPD8AAKC9AABEPwAAoL0AAEw/AACgvQAAVD8AAKC9AABcPwAAoL0AAGQ/AACgvQAAbD8AAKC9AAB0PwAAoL0AAHw/AACgvQAAfL8AAEC9AAB0vwAAQL0AAGy/AABAvQAAZL8AAEC9AABcvwAAQL0AAFS/AABAvQAATL8AAEC9AABEvwAAQL0AADy/AABAvQAANL8AAEC9AAAsvwAAQL0AACS/AABAvQAAHL8AAEC9AAAUvwAAQL0AAAy/AABAvQAABL8AAEC9AAD4vgAAQL0AAOi+AABAvQAA2L4AAEC9AADIvgAAQL0AALi+AABAvQAAqL4AAEC9AACYvgAAQL0AAIi+AABAvQAAcL4AAEC9AABQvgAAQL0AADC+AABAvQAAEL4AAEC9AADgvQAAQL0AAKC9AABAvQAAQL0AAEC9AACAvAAAQL0AAIA8AABAvQAAQD0AAEC9AACgPQAAQL0AAOA9AABAvQAAED4AAEC9AAAwPgAAQL0AAFA+AABAvQAAcD4AAEC9AACIPgAAQL0AAJg+AABAvQAAqD4AAEC9AAC4PgAAQL0AAMg+AABAvQAA2D4AAEC9AADoPgAAQL0AAPg+AABAvQAABD8AAEC9AAAMPwAAQL0AABQ/AABAvQAAHD8AAEC9AAAkPwAAQL0AACw/AABAvQAAND8AAEC9AAA8PwAAQL0AAEQ/AABAvQAATD8AAEC9AABUPwAAQL0AAFw/AABAvQAAZD8AAEC9AABsPwAAQL0AAHQ/AABAvQAAfD8AAEC9AAB8vwAAgLwAAHS/AACAvAAAbL8AAIC8AABkvwAAgLwAAFy/AACAvAAAVL8AAIC8AABMvwAAgLwAAES/AACAvAAAPL8AAIC8AAA0vwAAgLwAACy/AACAvAAAJL8AAIC8AAAcvwAAgLwAABS/AACAvAAADL8AAIC8AAAEvwAAgLwAAPi+AACAvAAA6L4AAIC8AADYvgAAgLwAAMi+AACAvAAAuL4AAIC8AACovgAAgLwAAJi+AACAvAAAiL4AAIC8AABwvgAAgLwAAFC+AACAvAAAML4AAIC8AAAQvgAAgLwAAOC9AACAvAAAoL0AAIC8AABAvQAAgLwAAIC8AACAvAAAgDwAAIC8AABAPQAAgLwAAKA9AACAvAAA4D0AAIC8AAAQPgAAgLwAADA+AACAvAAAUD4AAIC8AABwPgAAgLwAAIg+AACAvAAAmD4AAIC8AACoPgAAgLwAALg+AACAvAAAyD4AAIC8AADYPgAAgLwAAOg+AACAvAAA+D4AAIC8AAAEPwAAgLwAAAw/AACAvAAAFD8AAIC8AAAcPwAAgLwAACQ/AACAvAAALD8AAIC8AAA0PwAAgLwAADw/AACAvAAARD8AAIC8AABMPwAAgLwAAFQ/AACAvAAAXD8AAIC8AABkPwAAgLwAAGw/AACAvAAAdD8AAIC8AAB8PwAAgLwAAHy/AACAPAAAdL8AAIA8AABsvwAAgDwAAGS/AACAPAAAXL8AAIA8AABUvwAAgDwAAEy/AACAPAAARL8AAIA8AAA8vwAAgDwAADS/AACAPAAALL8AAIA8AAAkvwAAgDwAABy/AACAPAAAFL8AAIA8AAAMvwAAgDwAAAS/AACAPAAA+L4AAIA8AADovgAAgDwAANi+AACAPAAAyL4AAIA8AAC4vgAAgDwAAKi+AACAPAAAmL4AAIA8AACIvgAAgDwAAHC+AACAPAAAUL4AAIA8AAAwvgAAgDwAABC+AACAPAAA4L0AAIA8AACgvQAAgDwAAEC9AACAPAAAgLwAAIA8AACAPAAAgDwAAEA9AACAPAAAoD0AAIA8AADgPQAAgDwAABA+AACAPAAAMD4AAIA8AABQPgAAgDwAAHA+AACAPAAAiD4AAIA8AACYPgAAgDwAAKg+AACAPAAAuD4AAIA8AADIPgAAgDwAANg+AACAPAAA6D4AAIA8AAD4PgAAgDwAAAQ/AACAPAAADD8AAIA8AAAUPwAAgDwAABw/AACAPAAAJD8AAIA8AAAsPwAAgDwAADQ/AACAPAAAPD8AAIA8AABEPwAAgDwAAEw/AACAPAAAVD8AAIA8AABcPwAAgDwAAGQ/AACAPAAAbD8AAIA8AAB0PwAAgDwAAHw/AACAPAAAfL8AAEA9AAB0vwAAQD0AAGy/AABAPQAAZL8AAEA9AABcvwAAQD0AAFS/AABAPQAATL8AAEA9AABEvwAAQD0AADy/AABAPQAANL8AAEA9AAAsvwAAQD0AACS/AABAPQAAHL8AAEA9AAAUvwAAQD0AAAy/AABAPQAABL8AAEA9AAD4vgAAQD0AAOi+AABAPQAA2L4AAEA9AADIvgAAQD0AALi+AABAPQAAqL4AAEA9AACYvgAAQD0AAIi+AABAPQAAcL4AAEA9AABQvgAAQD0AADC+AABAPQAAEL4AAEA9AADgvQAAQD0AAKC9AABAPQAAQL0AAEA9AACAvAAAQD0AAIA8AABAPQAAQD0AAEA9AACgPQAAQD0AAOA9AABAPQAAED4AAEA9AAAwPgAAQD0AAFA+AABAPQAAcD4AAEA9AACIPgAAQD0AAJg+AABAPQAAqD4AAEA9AAC4PgAAQD0AAMg+AABAPQAA2D4AAEA9AADoPgAAQD0AAPg+AABAPQAABD8AAEA9AAAMPwAAQD0AABQ/AABAPQAAHD8AAEA9AAAkPwAAQD0AACw/AABAPQAAND8AAEA9AAA8PwAAQD0AAEQ/AABAPQAATD8AAEA9AABUPwAAQD0AAFw/AABAPQAAZD8AAEA9AABsPwAAQD0AAHQ/AABAPQAAfD8AAEA9AAB8vwAAoD0AAHS/AACgPQAAbL8AAKA9AABkvwAAoD0AAFy/AACgPQAAVL8AAKA9AABMvwAAoD0AAES/AACgPQAAPL8AAKA9AAA0vwAAoD0AACy/AACgPQAAJL8AAKA9AAAcvwAAoD0AABS/AACgPQAADL8AAKA9AAAEvwAAoD0AAPi+AACgPQAA6L4AAKA9AADYvgAAoD0AAMi+AACgPQAAuL4AAKA9AACovgAAoD0AAJi+AACgPQAAiL4AAKA9AABwvgAAoD0AAFC+AACgPQAAML4AAKA9AAAQvgAAoD0AAOC9AACgPQAAoL0AAKA9AABAvQAAoD0AAIC8AACgPQAAgDwAAKA9AABAPQAAoD0AAKA9AACgPQAA4D0AAKA9AAAQPgAAoD0AADA+AACgPQAAUD4AAKA9AABwPgAAoD0AAIg+AACgPQAAmD4AAKA9AACoPgAAoD0AALg+AACgPQAAyD4AAKA9AADYPgAAoD0AAOg+AACgPQAA+D4AAKA9AAAEPwAAoD0AAAw/AACgPQAAFD8AAKA9AAAcPwAAoD0AACQ/AACgPQAALD8AAKA9AAA0PwAAoD0AADw/AACgPQAARD8AAKA9AABMPwAAoD0AAFQ/AACgPQAAXD8AAKA9AABkPwAAoD0AAGw/AACgPQAAdD8AAKA9AAB8PwAAoD0AAHy/AADgPQAAdL8AAOA9AABsvwAA4D0AAGS/AADgPQAAXL8AAOA9AABUvwAA4D0AAEy/AADgPQAARL8AAOA9AAA8vwAA4D0AADS/AADgPQAALL8AAOA9AAAkvwAA4D0AABy/AADgPQAAFL8AAOA9AAAMvwAA4D0AAAS/AADgPQAA+L4AAOA9AADovgAA4D0AANi+AADgPQAAyL4AAOA9AAC4vgAA4D0AAKi+AADgPQAAmL4AAOA9AACIvgAA4D0AAHC+AADgPQAAUL4AAOA9AAAwvgAA4D0AABC+AADgPQAA4L0AAOA9AACgvQAA4D0AAEC9AADgPQAAgLwAAOA9AACAPAAA4D0AAEA9AADgPQAAoD0AAOA9AADgPQAA4D0AABA+AADgPQAAMD4AAOA9AABQPgAA4D0AAHA+AADgPQAAiD4AAOA9AACYPgAA4D0AAKg+AADgPQAAuD4AAOA9AADIPgAA4D0AANg+AADgPQAA6D4AAOA9AAD4PgAA4D0AAAQ/AADgPQAADD8AAOA9AAAUPwAA4D0AABw/AADgPQAAJD8AAOA9AAAsPwAA4D0AADQ/AADgPQAAPD8AAOA9AABEPwAA4D0AAEw/AADgPQAAVD8AAOA9AABcPwAA4D0AAGQ/AADgPQAAbD8AAOA9AAB0PwAA4D0AAHw/AADgPQAAfL8AABA+AAB0vwAAED4AAGy/AAAQPgAAZL8AABA+AABcvwAAED4AAFS/AAAQPgAATL8AABA+AABEvwAAED4AADy/AAAQPgAANL8AABA+AAAsvwAAED4AACS/AAAQPgAAHL8AABA+AAAUvwAAED4AAAy/AAAQPgAABL8AABA+AAD4vgAAED4AAOi+AAAQPgAA2L4AABA+AADIvgAAED4AALi+AAAQPgAAqL4AABA+AACYvgAAED4AAIi+AAAQPgAAcL4AABA+AABQvgAAED4AADC+AAAQPgAAEL4AABA+AADgvQAAED4AAKC9AAAQPgAAQL0AABA+AACAvAAAED4AAIA8AAAQPgAAQD0AABA+AACgPQAAED4AAOA9AAAQPgAAED4AABA+AAAwPgAAED4AAFA+AAAQPgAAcD4AABA+AACIPgAAED4AAJg+AAAQPgAAqD4AABA+AAC4PgAAED4AAMg+AAAQPgAA2D4AABA+AADoPgAAED4AAPg+AAAQPgAABD8AABA+AAAMPwAAED4AABQ/AAAQPgAAHD8AABA+AAAkPwAAED4AACw/AAAQPgAAND8AABA+AAA8PwAAED4AAEQ/AAAQPgAATD8AABA+AABUPwAAED4AAFw/AAAQPgAAZD8AABA+AABsPwAAED4AAHQ/AAAQPgAAfD8AABA+AAB8vwAAMD4AAHS/AAAwPgAAbL8AADA+AABkvwAAMD4AAFy/AAAwPgAAVL8AADA+AABMvwAAMD4AAES/AAAwPgAAPL8AADA+AAA0vwAAMD4AACy/AAAwPgAAJL8AADA+AAAcvwAAMD4AABS/AAAwPgAADL8AADA+AAAEvwAAMD4AAPi+AAAwPgAA6L4AADA+AADYvgAAMD4AAMi+AAAwPgAAuL4AADA+AACovgAAMD4AAJi+AAAwPgAAiL4AADA+AABwvgAAMD4AAFC+AAAwPgAAML4AADA+AAAQvgAAMD4AAOC9AAAwPgAAoL0AADA+AABAvQAAMD4AAIC8AAAwPgAAgDwAADA+AABAPQAAMD4AAKA9AAAwPgAA4D0AADA+AAAQPgAAMD4AADA+AAAwPgAAUD4AADA+AABwPgAAMD4AAIg+AAAwPgAAmD4AADA+AACoPgAAMD4AALg+AAAwPgAAyD4AADA+AADYPgAAMD4AAOg+AAAwPgAA+D4AADA+AAAEPwAAMD4AAAw/AAAwPgAAFD8AADA+AAAcPwAAMD4AACQ/AAAwPgAALD8AADA+AAA0PwAAMD4AADw/AAAwPgAARD8AADA+AABMPwAAMD4AAFQ/AAAwPgAAXD8AADA+AABkPwAAMD4AAGw/AAAwPgAAdD8AADA+AAB8PwAAMD4AAHy/AABQPgAAdL8AAFA+AABsvwAAUD4AAGS/AABQPgAAXL8AAFA+AABUvwAAUD4AAEy/AABQPgAARL8AAFA+AAA8vwAAUD4AADS/AABQPgAALL8AAFA+AAAkvwAAUD4AABy/AABQPgAAFL8AAFA+AAAMvwAAUD4AAAS/AABQPgAA+L4AAFA+AADovgAAUD4AANi+AABQPgAAyL4AAFA+AAC4vgAAUD4AAKi+AABQPgAAmL4AAFA+AACIvgAAUD4AAHC+AABQPgAAUL4AAFA+AAAwvgAAUD4AABC+AABQPgAA4L0AAFA+AACgvQAAUD4AAEC9AABQPgAAgLwAAFA+AACAPAAAUD4AAEA9AABQPgAAoD0AAFA+AADgPQAAUD4AABA+AABQPgAAMD4AAFA+AABQPgAAUD4AAHA+AABQPgAAiD4AAFA+AACYPgAAUD4AAKg+AABQPgAAuD4AAFA+AADIPgAAUD4AANg+AABQPgAA6D4AAFA+AAD4PgAAUD4AAAQ/AABQPgAADD8AAFA+AAAUPwAAUD4AABw/AABQPgAAJD8AAFA+AAAsPwAAUD4AADQ/AABQPgAAPD8AAFA+AABEPwAAUD4AAEw/AABQPgAAVD8AAFA+AABcPwAAUD4AAGQ/AABQPgAAbD8AAFA+AAB0PwAAUD4AAHw/AABQPgAAfL8AAHA+AAB0vwAAcD4AAGy/AABwPgAAZL8AAHA+AABcvwAAcD4AAFS/AABwPgAATL8AAHA+AABEvwAAcD4AADy/AABwPgAANL8AAHA+AAAsvwAAcD4AACS/AABwPgAAHL8AAHA+AAAUvwAAcD4AAAy/AABwPgAABL8AAHA+AAD4vgAAcD4AAOi+AABwPgAA2L4AAHA+AADIvgAAcD4AALi+AABwPgAAqL4AAHA+AACYvgAAcD4AAIi+AABwPgAAcL4AAHA+AABQvgAAcD4AADC+AABwPgAAEL4AAHA+AADgvQAAcD4AAKC9AABwPgAAQL0AAHA+AACAvAAAcD4AAIA8AABwPgAAQD0AAHA+AACgPQAAcD4AAOA9AABwPgAAED4AAHA+AAAwPgAAcD4AAFA+AABwPgAAcD4AAHA+AACIPgAAcD4AAJg+AABwPgAAqD4AAHA+AAC4PgAAcD4AAMg+AABwPgAA2D4AAHA+AADoPgAAcD4AAPg+AABwPgAABD8AAHA+AAAMPwAAcD4AABQ/AABwPgAAHD8AAHA+AAAkPwAAcD4AACw/AABwPgAAND8AAHA+AAA8PwAAcD4AAEQ/AABwPgAATD8AAHA+AABUPwAAcD4AAFw/AABwPgAAZD8AAHA+AABsPwAAcD4AAHQ/AABwPgAAfD8AAHA+AAB8vwAAiD4AAHS/AACIPgAAbL8AAIg+AABkvwAAiD4AAFy/AACIPgAAVL8AAIg+AABMvwAAiD4AAES/AACIPgAAPL8AAIg+AAA0vwAAiD4AACy/AACIPgAAJL8AAIg+AAAcvwAAiD4AABS/AACIPgAADL8AAIg+AAAEvwAAiD4AAPi+AACIPgAA6L4AAIg+AADYvgAAiD4AAMi+AACIPgAAuL4AAIg+AACovgAAiD4AAJi+AACIPgAAiL4AAIg+AABwvgAAiD4AAFC+AACIPgAAML4AAIg+AAAQvgAAiD4AAOC9AACIPgAAoL0AAIg+AABAvQAAiD4AAIC8AACIPgAAgDwAAIg+AABAPQAAiD4AAKA9AACIPgAA4D0AAIg+AAAQPgAAiD4AADA+AACIPgAAUD4AAIg+AABwPgAAiD4AAIg+AACIPgAAmD4AAIg+AACoPgAAiD4AALg+AACIPgAAyD4AAIg+AADYPgAAiD4AAOg+AACIPgAA+D4AAIg+AAAEPwAAiD4AAAw/AACIPgAAFD8AAIg+AAAcPwAAiD4AACQ/AACIPgAALD8AAIg+AAA0PwAAiD4AADw/AACIPgAARD8AAIg+AABMPwAAiD4AAFQ/AACIPgAAXD8AAIg+AABkPwAAiD4AAGw/AACIPgAAdD8AAIg+AAB8PwAAiD4AAHy/AACYPgAAdL8AAJg+AABsvwAAmD4AAGS/AACYPgAAXL8AAJg+AABUvwAAmD4AAEy/AACYPgAARL8AAJg+AAA8vwAAmD4AADS/AACYPgAALL8AAJg+AAAkvwAAmD4AABy/AACYPgAAFL8AAJg+AAAMvwAAmD4AAAS/AACYPgAA+L4AAJg+AADovgAAmD4AANi+AACYPgAAyL4AAJg+AAC4vgAAmD4AAKi+AACYPgAAmL4AAJg+AACIvgAAmD4AAHC+AACYPgAAUL4AAJg+AAAwvgAAmD4AABC+AACYPgAA4L0AAJg+AACgvQAAmD4AAEC9AACYPgAAgLwAAJg+AACAPAAAmD4AAEA9AACYPgAAoD0AAJg+AADgPQAAmD4AABA+AACYPgAAMD4AAJg+AABQPgAAmD4AAHA+AACYPgAAiD4AAJg+AACYPgAAmD4AAKg+AACYPgAAuD4AAJg+AADIPgAAmD4AANg+AACYPgAA6D4AAJg+AAD4PgAAmD4AAAQ/AACYPgAADD8AAJg+AAAUPwAAmD4AABw/AACYPgAAJD8AAJg+AAAsPwAAmD4AADQ/AACYPgAAPD8AAJg+AABEPwAAmD4AAEw/AACYPgAAVD8AAJg+AABcPwAAmD4AAGQ/AACYPgAAbD8AAJg+AAB0PwAAmD4AAHw/AACYPgAAfL8AAKg+AAB0vwAAqD4AAGy/AACoPgAAZL8AAKg+AABcvwAAqD4AAFS/AACoPgAATL8AAKg+AABEvwAAqD4AADy/AACoPgAANL8AAKg+AAAsvwAAqD4AACS/AACoPgAAHL8AAKg+AAAUvwAAqD4AAAy/AACoPgAABL8AAKg+AAD4vgAAqD4AAOi+AACoPgAA2L4AAKg+AADIvgAAqD4AALi+AACoPgAAqL4AAKg+AACYvgAAqD4AAIi+AACoPgAAcL4AAKg+AABQvgAAqD4AADC+AACoPgAAEL4AAKg+AADgvQAAqD4AAKC9AACoPgAAQL0AAKg+AACAvAAAqD4AAIA8AACoPgAAQD0AAKg+AACgPQAAqD4AAOA9AACoPgAAED4AAKg+AAAwPgAAqD4AAFA+AACoPgAAcD4AAKg+AACIPgAAqD4AAJg+AACoPgAAqD4AAKg+AAC4PgAAqD4AAMg+AACoPgAA2D4AAKg+AADoPgAAqD4AAPg+AACoPgAABD8AAKg+AAAMPwAAqD4AABQ/AACoPgAAHD8AAKg+AAAkPwAAqD4AACw/AACoPgAAND8AAKg+AAA8PwAAqD4AAEQ/AACoPgAATD8AAKg+AABUPwAAqD4AAFw/AACoPgAAZD8AAKg+AABsPwAAqD4AAHQ/AACoPgAAfD8AAKg+AAB8vwAAuD4AAHS/AAC4PgAAbL8AALg+AABkvwAAuD4AAFy/AAC4PgAAVL8AALg+AABMvwAAuD4AAES/AAC4PgAAPL8AALg+AAA0vwAAuD4AACy/AAC4PgAAJL8AALg+AAAcvwAAuD4AABS/AAC4PgAADL8AALg+AAAEvwAAuD4AAPi+AAC4PgAA6L4AALg+AADYvgAAuD4AAMi+AAC4PgAAuL4AALg+AACovgAAuD4AAJi+AAC4PgAAiL4AALg+AABwvgAAuD4AAFC+AAC4PgAAML4AALg+AAAQvgAAuD4AAOC9AAC4PgAAoL0AALg+AABAvQAAuD4AAIC8AAC4PgAAgDwAALg+AABAPQAAuD4AAKA9AAC4PgAA4D0AALg+AAAQPgAAuD4AADA+AAC4PgAAUD4AALg+AABwPgAAuD4AAIg+AAC4PgAAmD4AALg+AACoPgAAuD4AALg+AAC4PgAAyD4AALg+AADYPgAAuD4AAOg+AAC4PgAA+D4AALg+AAAEPwAAuD4AAAw/AAC4PgAAFD8AALg+AAAcPwAAuD4AACQ/AAC4PgAALD8AALg+AAA0PwAAuD4AADw/AAC4PgAARD8AALg+AABMPwAAuD4AAFQ/AAC4PgAAXD8AALg+AABkPwAAuD4AAGw/AAC4PgAAdD8AALg+AAB8PwAAuD4AAHy/AADIPgAAdL8AAMg+AABsvwAAyD4AAGS/AADIPgAAXL8AAMg+AABUvwAAyD4AAEy/AADIPgAARL8AAMg+AAA8vwAAyD4AADS/AADIPgAALL8AAMg+AAAkvwAAyD4AABy/AADIPgAAFL8AAMg+AAAMvwAAyD4AAAS/AADIPgAA+L4AAMg+AADovgAAyD4AANi+AADIPgAAyL4AAMg+AAC4vgAAyD4AAKi+AADIPgAAmL4AAMg+AACIvgAAyD4AAHC+AADIPgAAUL4AAMg+AAAwvgAAyD4AABC+AADIPgAA4L0AAMg+AACgvQAAyD4AAEC9AADIPgAAgLwAAMg+AACAPAAAyD4AAEA9AADIPgAAoD0AAMg+AADgPQAAyD4AABA+AADIPgAAMD4AAMg+AABQPgAAyD4AAHA+AADIPgAAiD4AAMg+AACYPgAAyD4AAKg+AADIPgAAuD4AAMg+AADIPgAAyD4AANg+AADIPgAA6D4AAMg+AAD4PgAAyD4AAAQ/AADIPgAADD8AAMg+AAAUPwAAyD4AABw/AADIPgAAJD8AAMg+AAAsPwAAyD4AADQ/AADIPgAAPD8AAMg+AABEPwAAyD4AAEw/AADIPgAAVD8AAMg+AABcPwAAyD4AAGQ/AADIPgAAbD8AAMg+AAB0PwAAyD4AAHw/AADIPgAAfL8AANg+AAB0vwAA2D4AAGy/AADYPgAAZL8AANg+AABcvwAA2D4AAFS/AADYPgAATL8AANg+AABEvwAA2D4AADy/AADYPgAANL8AANg+AAAsvwAA2D4AACS/AADYPgAAHL8AANg+AAAUvwAA2D4AAAy/AADYPgAABL8AANg+AAD4vgAA2D4AAOi+AADYPgAA2L4AANg+AADIvgAA2D4AALi+AADYPgAAqL4AANg+AACYvgAA2D4AAIi+AADYPgAAcL4AANg+AABQvgAA2D4AADC+AADYPgAAEL4AANg+AADgvQAA2D4AAKC9AADYPgAAQL0AANg+AACAvAAA2D4AAIA8AADYPgAAQD0AANg+AACgPQAA2D4AAOA9AADYPgAAED4AANg+AAAwPgAA2D4AAFA+AADYPgAAcD4AANg+AACIPgAA2D4AAJg+AADYPgAAqD4AANg+AAC4PgAA2D4AAMg+AADYPgAA2D4AANg+AADoPgAA2D4AAPg+AADYPgAABD8AANg+AAAMPwAA2D4AABQ/AADYPgAAHD8AANg+AAAkPwAA2D4AACw/AADYPgAAND8AANg+AAA8PwAA2D4AAEQ/AADYPgAATD8AANg+AABUPwAA2D4AAFw/AADYPgAAZD8AANg+AABsPwAA2D4AAHQ/AADYPgAAfD8AANg+AAB8vwAA6D4AAHS/AADoPgAAbL8AAOg+AABkvwAA6D4AAFy/AADoPgAAVL8AAOg+AABMvwAA6D4AAES/AADoPgAAPL8AAOg+AAA0vwAA6D4AACy/AADoPgAAJL8AAOg+AAAcvwAA6D4AABS/AADoPgAADL8AAOg+AAAEvwAA6D4AAPi+AADoPgAA6L4AAOg+AADYvgAA6D4AAMi+AADoPgAAuL4AAOg+AACovgAA6D4AAJi+AADoPgAAiL4AAOg+AABwvgAA6D4AAFC+AADoPgAAML4AAOg+AAAQvgAA6D4AAOC9AADoPgAAoL0AAOg+AABAvQAA6D4AAIC8AADoPgAAgDwAAOg+AABAPQAA6D4AAKA9AADoPgAA4D0AAOg+AAAQPgAA6D4AADA+AADoPgAAUD4AAOg+AABwPgAA6D4AAIg+AADoPgAAmD4AAOg+AACoPgAA6D4AALg+AADoPgAAyD4AAOg+AADYPgAA6D4AAOg+AADoPgAA+D4AAOg+AAAEPwAA6D4AAAw/AADoPgAAFD8AAOg+AAAcPwAA6D4AACQ/AADoPgAALD8AAOg+AAA0PwAA6D4AADw/AADoPgAARD8AAOg+AABMPwAA6D4AAFQ/AADoPgAAXD8AAOg+AABkPwAA6D4AAGw/AADoPgAAdD8AAOg+AAB8PwAA6D4AAHy/AAD4PgAAdL8AAPg+AABsvwAA+D4AAGS/AAD4PgAAXL8AAPg+AABUvwAA+D4AAEy/AAD4PgAARL8AAPg+AAA8vwAA+D4AADS/AAD4PgAALL8AAPg+AAAkvwAA+D4AABy/AAD4PgAAFL8AAPg+AAAMvwAA+D4AAAS/AAD4PgAA+L4AAPg+AADovgAA+D4AANi+AAD4PgAAyL4AAPg+AAC4vgAA+D4AAKi+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AAAsvwAAfD8AACS/AAB8PwAAHL8AAHw/AAAUvwAAfD8AAAy/AAB8PwAABL8AAHw/AAD4vgAAfD8AAOi+AAB8PwAA2L4AAHw/AADIvgAAfD8AALi+AAB8PwAAqL4AAHw/AACYvgAAfD8AAIi+AAB8PwAAcL4AAHw/AABQvgAAfD8AADC+AAB8PwAAEL4AAHw/AADgvQAAfD8AAKC9AAB8PwAAQL0AAHw/AACAvAAAfD8AAIA8AAB8PwAAQD0AAHw/AACgPQAAfD8AAOA9AAB8PwAAED4AAHw/AAAwPgAAfD8AAFA+AAB8PwAAcD4AAHw/AACIPgAAfD8AAJg+AAB8PwAAqD4AAHw/AAC4PgAAfD8AAMg+AAB8PwAA2D4AAHw/AADoPgAAfD8AAPg+AAB8PwAABD8AAHw/AAAMPwAAfD8AABQ/AAB8PwAAHD8AAHw/AAAkPwAAfD8AACw/AAB8PwAAND8AAHw/AAA8PwAAfD8AAEQ/AAB8PwAATD8AAHw/AABUPwAAfD8AAFw/AAB8PwAAZD8AAHw/AABsPwAAfD8AAHQ/AAB8PwAAfD8AAHw/\", 
    \"P6G0lvBAXt0AAAAAAAAAAAEAAAAAAAAABAAAAAIgAQABAAAAAAAAAAEAAAAAAAAAAQAAAAAAAAACAAAAAAAAAAgAAAAAAAAAAAAAAAAAAAA=\", 
    \"P6G0lvBAXt0AAAAAAAAAAAEAAAAAAAAABAAAAAEBAQABAAAAAAAAAAEAAAAAAAAAAgAAAAAAAAABAAAAAAAAAAIAAAAAAAAAAAE=\", 
    \"P6G0lvBAXt0AAAAAAAAAAAEAAAAAAAAABAAAAAEBAQABAAAAAAAAAAEAAAAAAAAAAgAAAAAAAAABAAAAAAAAAAIAAAAAAAAAAQE=\", 
    \"P6G0lvBAXt0AAAAAAAAAAAEAAAAAAAAABAAAAAEBAQABAAAAAAAAAAEAAAAAAAAAAgAAAAAAAAABAAAAAAAAAAIAAAAAAAAAAAA=\", 
    \"P6G0lvBAXt0AAAAAAAAAAAEAAAAAAAAABAAAAAEBAQABAAAAAAAAAAEAAAAAAAAAAgAAAAAAAAABAAAAAAAAAAIAAAAAAAAAAQA=\"
  ], 
  \"attrs\": {\"tvm_version\": \"0.15.dev0\"}
}""")

@I.ir_module
class SAMModuleDebug:
    @R.function
    def _initialize_effect() -> R.Tuple(R.Object):
        with R.dataflow():
            _io: R.Object = R.null_value()
            gv: R.Tuple(R.Object) = (_io,)
            R.output(gv)
        return gv

    @R.function
    def main(pixel_values: R.Tensor((1, 3, 1024, 1024), dtype="float32"), input_points: R.Tensor((1, 1, 1, 2), dtype="float32"), _io: R.Object, shared_image_embedding_positional_embedding: R.Tensor((2, 128), dtype="float32"), vision_encoder_patch_embed_projection_weight: R.Tensor((768, 3, 16, 16), dtype="float32"), vision_encoder_patch_embed_projection_bias: R.Tensor((768,), dtype="float32"), vision_encoder_pos_embed: R.Tensor((1, 64, 64, 768), dtype="float32"), vision_encoder_layers_0_layer_norm1_weight: R.Tensor((768,), dtype="float32"), vision_encoder_layers_0_layer_norm1_bias: R.Tensor((768,), dtype="float32"), vision_encoder_layers_0_attn_qkv_weight: R.Tensor((2304, 768), dtype="float32"), vision_encoder_layers_0_attn_qkv_bias: R.Tensor((2304,), dtype="float32"), vision_encoder_layers_0_attn_proj_weight: R.Tensor((768, 768), dtype="float32"), vision_encoder_layers_0_attn_proj_bias: R.Tensor((768,), dtype="float32"), vision_encoder_layers_0_attn_rel_pos_h: R.Tensor((27, 64), dtype="float32"), vision_encoder_layers_0_attn_rel_pos_w: R.Tensor((27, 64), dtype="float32"), vision_encoder_layers_0_layer_norm2_weight: R.Tensor((768,), dtype="float32"), vision_encoder_layers_0_layer_norm2_bias: R.Tensor((768,), dtype="float32"), vision_encoder_layers_0_mlp_lin1_weight: R.Tensor((3072, 768), dtype="float32"), vision_encoder_layers_0_mlp_lin1_bias: R.Tensor((3072,), dtype="float32"), vision_encoder_layers_0_mlp_lin2_weight: R.Tensor((768, 3072), dtype="float32"), vision_encoder_layers_0_mlp_lin2_bias: R.Tensor((768,), dtype="float32"), vision_encoder_layers_1_layer_norm1_weight: R.Tensor((768,), dtype="float32"), vision_encoder_layers_1_layer_norm1_bias: R.Tensor((768,), dtype="float32"), vision_encoder_layers_1_attn_qkv_weight: R.Tensor((2304, 768), dtype="float32"), vision_encoder_layers_1_attn_qkv_bias: R.Tensor((2304,), dtype="float32"), vision_encoder_layers_1_attn_proj_weight: R.Tensor((768, 768), dtype="float32"), vision_encoder_layers_1_attn_proj_bias: R.Tensor((768,), dtype="float32"), vision_encoder_layers_1_attn_rel_pos_h: R.Tensor((27, 64), dtype="float32"), vision_encoder_layers_1_attn_rel_pos_w: R.Tensor((27, 64), dtype="float32"), vision_encoder_layers_1_layer_norm2_weight: R.Tensor((768,), dtype="float32"), vision_encoder_layers_1_layer_norm2_bias: R.Tensor((768,), dtype="float32"), vision_encoder_layers_1_mlp_lin1_weight: R.Tensor((3072, 768), dtype="float32"), vision_encoder_layers_1_mlp_lin1_bias: R.Tensor((3072,), dtype="float32"), vision_encoder_layers_1_mlp_lin2_weight: R.Tensor((768, 3072), dtype="float32"), vision_encoder_layers_1_mlp_lin2_bias: R.Tensor((768,), dtype="float32"), vision_encoder_layers_2_layer_norm1_weight: R.Tensor((768,), dtype="float32"), vision_encoder_layers_2_layer_norm1_bias: R.Tensor((768,), dtype="float32"), vision_encoder_layers_2_attn_qkv_weight: R.Tensor((2304, 768), dtype="float32"), vision_encoder_layers_2_attn_qkv_bias: R.Tensor((2304,), dtype="float32"), vision_encoder_layers_2_attn_proj_weight: R.Tensor((768, 768), dtype="float32"), vision_encoder_layers_2_attn_proj_bias: R.Tensor((768,), dtype="float32"), vision_encoder_layers_2_attn_rel_pos_h: R.Tensor((127, 64), dtype="float32"), vision_encoder_layers_2_attn_rel_pos_w: R.Tensor((127, 64), dtype="float32"), vision_encoder_layers_2_layer_norm2_weight: R.Tensor((768,), dtype="float32"), vision_encoder_layers_2_layer_norm2_bias: R.Tensor((768,), dtype="float32"), vision_encoder_layers_2_mlp_lin1_weight: R.Tensor((3072, 768), dtype="float32"), vision_encoder_layers_2_mlp_lin1_bias: R.Tensor((3072,), dtype="float32"), vision_encoder_layers_2_mlp_lin2_weight: R.Tensor((768, 3072), dtype="float32"), vision_encoder_layers_2_mlp_lin2_bias: R.Tensor((768,), dtype="float32"), vision_encoder_layers_3_layer_norm1_weight: R.Tensor((768,), dtype="float32"), vision_encoder_layers_3_layer_norm1_bias: R.Tensor((768,), dtype="float32"), vision_encoder_layers_3_attn_qkv_weight: R.Tensor((2304, 768), dtype="float32"), vision_encoder_layers_3_attn_qkv_bias: R.Tensor((2304,), dtype="float32"), vision_encoder_layers_3_attn_proj_weight: R.Tensor((768, 768), dtype="float32"), vision_encoder_layers_3_attn_proj_bias: R.Tensor((768,), dtype="float32"), vision_encoder_layers_3_attn_rel_pos_h: R.Tensor((27, 64), dtype="float32"), vision_encoder_layers_3_attn_rel_pos_w: R.Tensor((27, 64), dtype="float32"), vision_encoder_layers_3_layer_norm2_weight: R.Tensor((768,), dtype="float32"), vision_encoder_layers_3_layer_norm2_bias: R.Tensor((768,), dtype="float32"), vision_encoder_layers_3_mlp_lin1_weight: R.Tensor((3072, 768), dtype="float32"), vision_encoder_layers_3_mlp_lin1_bias: R.Tensor((3072,), dtype="float32"), vision_encoder_layers_3_mlp_lin2_weight: R.Tensor((768, 3072), dtype="float32"), vision_encoder_layers_3_mlp_lin2_bias: R.Tensor((768,), dtype="float32"), vision_encoder_layers_4_layer_norm1_weight: R.Tensor((768,), dtype="float32"), vision_encoder_layers_4_layer_norm1_bias: R.Tensor((768,), dtype="float32"), vision_encoder_layers_4_attn_qkv_weight: R.Tensor((2304, 768), dtype="float32"), vision_encoder_layers_4_attn_qkv_bias: R.Tensor((2304,), dtype="float32"), vision_encoder_layers_4_attn_proj_weight: R.Tensor((768, 768), dtype="float32"), vision_encoder_layers_4_attn_proj_bias: R.Tensor((768,), dtype="float32"), vision_encoder_layers_4_attn_rel_pos_h: R.Tensor((27, 64), dtype="float32"), vision_encoder_layers_4_attn_rel_pos_w: R.Tensor((27, 64), dtype="float32"), vision_encoder_layers_4_layer_norm2_weight: R.Tensor((768,), dtype="float32"), vision_encoder_layers_4_layer_norm2_bias: R.Tensor((768,), dtype="float32"), vision_encoder_layers_4_mlp_lin1_weight: R.Tensor((3072, 768), dtype="float32"), vision_encoder_layers_4_mlp_lin1_bias: R.Tensor((3072,), dtype="float32"), vision_encoder_layers_4_mlp_lin2_weight: R.Tensor((768, 3072), dtype="float32"), vision_encoder_layers_4_mlp_lin2_bias: R.Tensor((768,), dtype="float32"), vision_encoder_layers_5_layer_norm1_weight: R.Tensor((768,), dtype="float32"), vision_encoder_layers_5_layer_norm1_bias: R.Tensor((768,), dtype="float32"), vision_encoder_layers_5_attn_qkv_weight: R.Tensor((2304, 768), dtype="float32"), vision_encoder_layers_5_attn_qkv_bias: R.Tensor((2304,), dtype="float32"), vision_encoder_layers_5_attn_proj_weight: R.Tensor((768, 768), dtype="float32"), vision_encoder_layers_5_attn_proj_bias: R.Tensor((768,), dtype="float32"), vision_encoder_layers_5_attn_rel_pos_h: R.Tensor((127, 64), dtype="float32"), vision_encoder_layers_5_attn_rel_pos_w: R.Tensor((127, 64), dtype="float32"), vision_encoder_layers_5_layer_norm2_weight: R.Tensor((768,), dtype="float32"), vision_encoder_layers_5_layer_norm2_bias: R.Tensor((768,), dtype="float32"), vision_encoder_layers_5_mlp_lin1_weight: R.Tensor((3072, 768), dtype="float32"), vision_encoder_layers_5_mlp_lin1_bias: R.Tensor((3072,), dtype="float32"), vision_encoder_layers_5_mlp_lin2_weight: R.Tensor((768, 3072), dtype="float32"), vision_encoder_layers_5_mlp_lin2_bias: R.Tensor((768,), dtype="float32"), vision_encoder_layers_6_layer_norm1_weight: R.Tensor((768,), dtype="float32"), vision_encoder_layers_6_layer_norm1_bias: R.Tensor((768,), dtype="float32"), vision_encoder_layers_6_attn_qkv_weight: R.Tensor((2304, 768), dtype="float32"), vision_encoder_layers_6_attn_qkv_bias: R.Tensor((2304,), dtype="float32"), vision_encoder_layers_6_attn_proj_weight: R.Tensor((768, 768), dtype="float32"), vision_encoder_layers_6_attn_proj_bias: R.Tensor((768,), dtype="float32"), vision_encoder_layers_6_attn_rel_pos_h: R.Tensor((27, 64), dtype="float32"), vision_encoder_layers_6_attn_rel_pos_w: R.Tensor((27, 64), dtype="float32"), vision_encoder_layers_6_layer_norm2_weight: R.Tensor((768,), dtype="float32"), vision_encoder_layers_6_layer_norm2_bias: R.Tensor((768,), dtype="float32"), vision_encoder_layers_6_mlp_lin1_weight: R.Tensor((3072, 768), dtype="float32"), vision_encoder_layers_6_mlp_lin1_bias: R.Tensor((3072,), dtype="float32"), vision_encoder_layers_6_mlp_lin2_weight: R.Tensor((768, 3072), dtype="float32"), vision_encoder_layers_6_mlp_lin2_bias: R.Tensor((768,), dtype="float32"), vision_encoder_layers_7_layer_norm1_weight: R.Tensor((768,), dtype="float32"), vision_encoder_layers_7_layer_norm1_bias: R.Tensor((768,), dtype="float32"), vision_encoder_layers_7_attn_qkv_weight: R.Tensor((2304, 768), dtype="float32"), vision_encoder_layers_7_attn_qkv_bias: R.Tensor((2304,), dtype="float32"), vision_encoder_layers_7_attn_proj_weight: R.Tensor((768, 768), dtype="float32"), vision_encoder_layers_7_attn_proj_bias: R.Tensor((768,), dtype="float32"), vision_encoder_layers_7_attn_rel_pos_h: R.Tensor((27, 64), dtype="float32"), vision_encoder_layers_7_attn_rel_pos_w: R.Tensor((27, 64), dtype="float32"), vision_encoder_layers_7_layer_norm2_weight: R.Tensor((768,), dtype="float32"), vision_encoder_layers_7_layer_norm2_bias: R.Tensor((768,), dtype="float32"), vision_encoder_layers_7_mlp_lin1_weight: R.Tensor((3072, 768), dtype="float32"), vision_encoder_layers_7_mlp_lin1_bias: R.Tensor((3072,), dtype="float32"), vision_encoder_layers_7_mlp_lin2_weight: R.Tensor((768, 3072), dtype="float32"), vision_encoder_layers_7_mlp_lin2_bias: R.Tensor((768,), dtype="float32"), vision_encoder_layers_8_layer_norm1_weight: R.Tensor((768,), dtype="float32"), vision_encoder_layers_8_layer_norm1_bias: R.Tensor((768,), dtype="float32"), vision_encoder_layers_8_attn_qkv_weight: R.Tensor((2304, 768), dtype="float32"), vision_encoder_layers_8_attn_qkv_bias: R.Tensor((2304,), dtype="float32"), vision_encoder_layers_8_attn_proj_weight: R.Tensor((768, 768), dtype="float32"), vision_encoder_layers_8_attn_proj_bias: R.Tensor((768,), dtype="float32"), vision_encoder_layers_8_attn_rel_pos_h: R.Tensor((127, 64), dtype="float32"), vision_encoder_layers_8_attn_rel_pos_w: R.Tensor((127, 64), dtype="float32"), vision_encoder_layers_8_layer_norm2_weight: R.Tensor((768,), dtype="float32"), vision_encoder_layers_8_layer_norm2_bias: R.Tensor((768,), dtype="float32"), vision_encoder_layers_8_mlp_lin1_weight: R.Tensor((3072, 768), dtype="float32"), vision_encoder_layers_8_mlp_lin1_bias: R.Tensor((3072,), dtype="float32"), vision_encoder_layers_8_mlp_lin2_weight: R.Tensor((768, 3072), dtype="float32"), vision_encoder_layers_8_mlp_lin2_bias: R.Tensor((768,), dtype="float32"), vision_encoder_layers_9_layer_norm1_weight: R.Tensor((768,), dtype="float32"), vision_encoder_layers_9_layer_norm1_bias: R.Tensor((768,), dtype="float32"), vision_encoder_layers_9_attn_qkv_weight: R.Tensor((2304, 768), dtype="float32"), vision_encoder_layers_9_attn_qkv_bias: R.Tensor((2304,), dtype="float32"), vision_encoder_layers_9_attn_proj_weight: R.Tensor((768, 768), dtype="float32"), vision_encoder_layers_9_attn_proj_bias: R.Tensor((768,), dtype="float32"), vision_encoder_layers_9_attn_rel_pos_h: R.Tensor((27, 64), dtype="float32"), vision_encoder_layers_9_attn_rel_pos_w: R.Tensor((27, 64), dtype="float32"), vision_encoder_layers_9_layer_norm2_weight: R.Tensor((768,), dtype="float32"), vision_encoder_layers_9_layer_norm2_bias: R.Tensor((768,), dtype="float32"), vision_encoder_layers_9_mlp_lin1_weight: R.Tensor((3072, 768), dtype="float32"), vision_encoder_layers_9_mlp_lin1_bias: R.Tensor((3072,), dtype="float32"), vision_encoder_layers_9_mlp_lin2_weight: R.Tensor((768, 3072), dtype="float32"), vision_encoder_layers_9_mlp_lin2_bias: R.Tensor((768,), dtype="float32"), vision_encoder_layers_10_layer_norm1_weight: R.Tensor((768,), dtype="float32"), vision_encoder_layers_10_layer_norm1_bias: R.Tensor((768,), dtype="float32"), vision_encoder_layers_10_attn_qkv_weight: R.Tensor((2304, 768), dtype="float32"), vision_encoder_layers_10_attn_qkv_bias: R.Tensor((2304,), dtype="float32"), vision_encoder_layers_10_attn_proj_weight: R.Tensor((768, 768), dtype="float32"), vision_encoder_layers_10_attn_proj_bias: R.Tensor((768,), dtype="float32"), vision_encoder_layers_10_attn_rel_pos_h: R.Tensor((27, 64), dtype="float32"), vision_encoder_layers_10_attn_rel_pos_w: R.Tensor((27, 64), dtype="float32"), vision_encoder_layers_10_layer_norm2_weight: R.Tensor((768,), dtype="float32"), vision_encoder_layers_10_layer_norm2_bias: R.Tensor((768,), dtype="float32"), vision_encoder_layers_10_mlp_lin1_weight: R.Tensor((3072, 768), dtype="float32"), vision_encoder_layers_10_mlp_lin1_bias: R.Tensor((3072,), dtype="float32"), vision_encoder_layers_10_mlp_lin2_weight: R.Tensor((768, 3072), dtype="float32"), vision_encoder_layers_10_mlp_lin2_bias: R.Tensor((768,), dtype="float32"), vision_encoder_layers_11_layer_norm1_weight: R.Tensor((768,), dtype="float32"), vision_encoder_layers_11_layer_norm1_bias: R.Tensor((768,), dtype="float32"), vision_encoder_layers_11_attn_qkv_weight: R.Tensor((2304, 768), dtype="float32"), vision_encoder_layers_11_attn_qkv_bias: R.Tensor((2304,), dtype="float32"), vision_encoder_layers_11_attn_proj_weight: R.Tensor((768, 768), dtype="float32"), vision_encoder_layers_11_attn_proj_bias: R.Tensor((768,), dtype="float32"), vision_encoder_layers_11_attn_rel_pos_h: R.Tensor((127, 64), dtype="float32"), vision_encoder_layers_11_attn_rel_pos_w: R.Tensor((127, 64), dtype="float32"), vision_encoder_layers_11_layer_norm2_weight: R.Tensor((768,), dtype="float32"), vision_encoder_layers_11_layer_norm2_bias: R.Tensor((768,), dtype="float32"), vision_encoder_layers_11_mlp_lin1_weight: R.Tensor((3072, 768), dtype="float32"), vision_encoder_layers_11_mlp_lin1_bias: R.Tensor((3072,), dtype="float32"), vision_encoder_layers_11_mlp_lin2_weight: R.Tensor((768, 3072), dtype="float32"), vision_encoder_layers_11_mlp_lin2_bias: R.Tensor((768,), dtype="float32"), vision_encoder_neck_conv1_weight: R.Tensor((256, 768, 1, 1), dtype="float32"), vision_encoder_neck_layer_norm1_weight: R.Tensor((256,), dtype="float32"), vision_encoder_neck_layer_norm1_bias: R.Tensor((256,), dtype="float32"), vision_encoder_neck_conv2_weight: R.Tensor((256, 256, 3, 3), dtype="float32"), vision_encoder_neck_layer_norm2_weight: R.Tensor((256,), dtype="float32"), vision_encoder_neck_layer_norm2_bias: R.Tensor((256,), dtype="float32"), prompt_encoder_shared_embedding_positional_embedding: R.Tensor((2, 128), dtype="float32"), prompt_encoder_mask_embed_conv1_weight: R.Tensor((4, 1, 2, 2), dtype="float32"), prompt_encoder_mask_embed_conv1_bias: R.Tensor((4,), dtype="float32"), prompt_encoder_mask_embed_conv2_weight: R.Tensor((16, 4, 2, 2), dtype="float32"), prompt_encoder_mask_embed_conv2_bias: R.Tensor((16,), dtype="float32"), prompt_encoder_mask_embed_conv3_weight: R.Tensor((256, 16, 1, 1), dtype="float32"), prompt_encoder_mask_embed_conv3_bias: R.Tensor((256,), dtype="float32"), prompt_encoder_mask_embed_layer_norm1_weight: R.Tensor((4,), dtype="float32"), prompt_encoder_mask_embed_layer_norm1_bias: R.Tensor((4,), dtype="float32"), prompt_encoder_mask_embed_layer_norm2_weight: R.Tensor((16,), dtype="float32"), prompt_encoder_mask_embed_layer_norm2_bias: R.Tensor((16,), dtype="float32"), prompt_encoder_no_mask_embed_weight: R.Tensor((1, 256), dtype="float32"), prompt_encoder_point_embed_0_weight: R.Tensor((1, 256), dtype="float32"), prompt_encoder_point_embed_1_weight: R.Tensor((1, 256), dtype="float32"), prompt_encoder_point_embed_2_weight: R.Tensor((1, 256), dtype="float32"), prompt_encoder_point_embed_3_weight: R.Tensor((1, 256), dtype="float32"), prompt_encoder_not_a_point_embed_weight: R.Tensor((1, 256), dtype="float32"), mask_decoder_iou_token_weight: R.Tensor((1, 256), dtype="float32"), mask_decoder_mask_tokens_weight: R.Tensor((4, 256), dtype="float32"), mask_decoder_transformer_layers_0_self_attn_q_proj_weight: R.Tensor((256, 256), dtype="float32"), mask_decoder_transformer_layers_0_self_attn_q_proj_bias: R.Tensor((256,), dtype="float32"), mask_decoder_transformer_layers_0_self_attn_k_proj_weight: R.Tensor((256, 256), dtype="float32"), mask_decoder_transformer_layers_0_self_attn_k_proj_bias: R.Tensor((256,), dtype="float32"), mask_decoder_transformer_layers_0_self_attn_v_proj_weight: R.Tensor((256, 256), dtype="float32"), mask_decoder_transformer_layers_0_self_attn_v_proj_bias: R.Tensor((256,), dtype="float32"), mask_decoder_transformer_layers_0_self_attn_out_proj_weight: R.Tensor((256, 256), dtype="float32"), mask_decoder_transformer_layers_0_self_attn_out_proj_bias: R.Tensor((256,), dtype="float32"), mask_decoder_transformer_layers_0_layer_norm1_weight: R.Tensor((256,), dtype="float32"), mask_decoder_transformer_layers_0_layer_norm1_bias: R.Tensor((256,), dtype="float32"), mask_decoder_transformer_layers_0_cross_attn_token_to_image_q_proj_weight: R.Tensor((128, 256), dtype="float32"), mask_decoder_transformer_layers_0_cross_attn_token_to_image_q_proj_bias: R.Tensor((128,), dtype="float32"), mask_decoder_transformer_layers_0_cross_attn_token_to_image_k_proj_weight: R.Tensor((128, 256), dtype="float32"), mask_decoder_transformer_layers_0_cross_attn_token_to_image_k_proj_bias: R.Tensor((128,), dtype="float32"), mask_decoder_transformer_layers_0_cross_attn_token_to_image_v_proj_weight: R.Tensor((128, 256), dtype="float32"), mask_decoder_transformer_layers_0_cross_attn_token_to_image_v_proj_bias: R.Tensor((128,), dtype="float32"), mask_decoder_transformer_layers_0_cross_attn_token_to_image_out_proj_weight: R.Tensor((256, 128), dtype="float32"), mask_decoder_transformer_layers_0_cross_attn_token_to_image_out_proj_bias: R.Tensor((256,), dtype="float32"), mask_decoder_transformer_layers_0_layer_norm2_weight: R.Tensor((256,), dtype="float32"), mask_decoder_transformer_layers_0_layer_norm2_bias: R.Tensor((256,), dtype="float32"), mask_decoder_transformer_layers_0_mlp_lin1_weight: R.Tensor((2048, 256), dtype="float32"), mask_decoder_transformer_layers_0_mlp_lin1_bias: R.Tensor((2048,), dtype="float32"), mask_decoder_transformer_layers_0_mlp_lin2_weight: R.Tensor((256, 2048), dtype="float32"), mask_decoder_transformer_layers_0_mlp_lin2_bias: R.Tensor((256,), dtype="float32"), mask_decoder_transformer_layers_0_layer_norm3_weight: R.Tensor((256,), dtype="float32"), mask_decoder_transformer_layers_0_layer_norm3_bias: R.Tensor((256,), dtype="float32"), mask_decoder_transformer_layers_0_layer_norm4_weight: R.Tensor((256,), dtype="float32"), mask_decoder_transformer_layers_0_layer_norm4_bias: R.Tensor((256,), dtype="float32"), mask_decoder_transformer_layers_0_cross_attn_image_to_token_q_proj_weight: R.Tensor((128, 256), dtype="float32"), mask_decoder_transformer_layers_0_cross_attn_image_to_token_q_proj_bias: R.Tensor((128,), dtype="float32"), mask_decoder_transformer_layers_0_cross_attn_image_to_token_k_proj_weight: R.Tensor((128, 256), dtype="float32"), mask_decoder_transformer_layers_0_cross_attn_image_to_token_k_proj_bias: R.Tensor((128,), dtype="float32"), mask_decoder_transformer_layers_0_cross_attn_image_to_token_v_proj_weight: R.Tensor((128, 256), dtype="float32"), mask_decoder_transformer_layers_0_cross_attn_image_to_token_v_proj_bias: R.Tensor((128,), dtype="float32"), mask_decoder_transformer_layers_0_cross_attn_image_to_token_out_proj_weight: R.Tensor((256, 128), dtype="float32"), mask_decoder_transformer_layers_0_cross_attn_image_to_token_out_proj_bias: R.Tensor((256,), dtype="float32"), mask_decoder_transformer_layers_1_self_attn_q_proj_weight: R.Tensor((256, 256), dtype="float32"), mask_decoder_transformer_layers_1_self_attn_q_proj_bias: R.Tensor((256,), dtype="float32"), mask_decoder_transformer_layers_1_self_attn_k_proj_weight: R.Tensor((256, 256), dtype="float32"), mask_decoder_transformer_layers_1_self_attn_k_proj_bias: R.Tensor((256,), dtype="float32"), mask_decoder_transformer_layers_1_self_attn_v_proj_weight: R.Tensor((256, 256), dtype="float32"), mask_decoder_transformer_layers_1_self_attn_v_proj_bias: R.Tensor((256,), dtype="float32"), mask_decoder_transformer_layers_1_self_attn_out_proj_weight: R.Tensor((256, 256), dtype="float32"), mask_decoder_transformer_layers_1_self_attn_out_proj_bias: R.Tensor((256,), dtype="float32"), mask_decoder_transformer_layers_1_layer_norm1_weight: R.Tensor((256,), dtype="float32"), mask_decoder_transformer_layers_1_layer_norm1_bias: R.Tensor((256,), dtype="float32"), mask_decoder_transformer_layers_1_cross_attn_token_to_image_q_proj_weight: R.Tensor((128, 256), dtype="float32"), mask_decoder_transformer_layers_1_cross_attn_token_to_image_q_proj_bias: R.Tensor((128,), dtype="float32"), mask_decoder_transformer_layers_1_cross_attn_token_to_image_k_proj_weight: R.Tensor((128, 256), dtype="float32"), mask_decoder_transformer_layers_1_cross_attn_token_to_image_k_proj_bias: R.Tensor((128,), dtype="float32"), mask_decoder_transformer_layers_1_cross_attn_token_to_image_v_proj_weight: R.Tensor((128, 256), dtype="float32"), mask_decoder_transformer_layers_1_cross_attn_token_to_image_v_proj_bias: R.Tensor((128,), dtype="float32"), mask_decoder_transformer_layers_1_cross_attn_token_to_image_out_proj_weight: R.Tensor((256, 128), dtype="float32"), mask_decoder_transformer_layers_1_cross_attn_token_to_image_out_proj_bias: R.Tensor((256,), dtype="float32"), mask_decoder_transformer_layers_1_layer_norm2_weight: R.Tensor((256,), dtype="float32"), mask_decoder_transformer_layers_1_layer_norm2_bias: R.Tensor((256,), dtype="float32"), mask_decoder_transformer_layers_1_mlp_lin1_weight: R.Tensor((2048, 256), dtype="float32"), mask_decoder_transformer_layers_1_mlp_lin1_bias: R.Tensor((2048,), dtype="float32"), mask_decoder_transformer_layers_1_mlp_lin2_weight: R.Tensor((256, 2048), dtype="float32"), mask_decoder_transformer_layers_1_mlp_lin2_bias: R.Tensor((256,), dtype="float32"), mask_decoder_transformer_layers_1_layer_norm3_weight: R.Tensor((256,), dtype="float32"), mask_decoder_transformer_layers_1_layer_norm3_bias: R.Tensor((256,), dtype="float32"), mask_decoder_transformer_layers_1_layer_norm4_weight: R.Tensor((256,), dtype="float32"), mask_decoder_transformer_layers_1_layer_norm4_bias: R.Tensor((256,), dtype="float32"), mask_decoder_transformer_layers_1_cross_attn_image_to_token_q_proj_weight: R.Tensor((128, 256), dtype="float32"), mask_decoder_transformer_layers_1_cross_attn_image_to_token_q_proj_bias: R.Tensor((128,), dtype="float32"), mask_decoder_transformer_layers_1_cross_attn_image_to_token_k_proj_weight: R.Tensor((128, 256), dtype="float32"), mask_decoder_transformer_layers_1_cross_attn_image_to_token_k_proj_bias: R.Tensor((128,), dtype="float32"), mask_decoder_transformer_layers_1_cross_attn_image_to_token_v_proj_weight: R.Tensor((128, 256), dtype="float32"), mask_decoder_transformer_layers_1_cross_attn_image_to_token_v_proj_bias: R.Tensor((128,), dtype="float32"), mask_decoder_transformer_layers_1_cross_attn_image_to_token_out_proj_weight: R.Tensor((256, 128), dtype="float32"), mask_decoder_transformer_layers_1_cross_attn_image_to_token_out_proj_bias: R.Tensor((256,), dtype="float32"), mask_decoder_transformer_final_attn_token_to_image_q_proj_weight: R.Tensor((128, 256), dtype="float32"), mask_decoder_transformer_final_attn_token_to_image_q_proj_bias: R.Tensor((128,), dtype="float32"), mask_decoder_transformer_final_attn_token_to_image_k_proj_weight: R.Tensor((128, 256), dtype="float32"), mask_decoder_transformer_final_attn_token_to_image_k_proj_bias: R.Tensor((128,), dtype="float32"), mask_decoder_transformer_final_attn_token_to_image_v_proj_weight: R.Tensor((128, 256), dtype="float32"), mask_decoder_transformer_final_attn_token_to_image_v_proj_bias: R.Tensor((128,), dtype="float32"), mask_decoder_transformer_final_attn_token_to_image_out_proj_weight: R.Tensor((256, 128), dtype="float32"), mask_decoder_transformer_final_attn_token_to_image_out_proj_bias: R.Tensor((256,), dtype="float32"), mask_decoder_transformer_layer_norm_final_attn_weight: R.Tensor((256,), dtype="float32"), mask_decoder_transformer_layer_norm_final_attn_bias: R.Tensor((256,), dtype="float32"), mask_decoder_upscale_conv1_weight: R.Tensor((256, 64, 2, 2), dtype="float32"), mask_decoder_upscale_conv1_bias: R.Tensor((64,), dtype="float32"), mask_decoder_upscale_conv2_weight: R.Tensor((64, 32, 2, 2), dtype="float32"), mask_decoder_upscale_conv2_bias: R.Tensor((32,), dtype="float32"), mask_decoder_upscale_layer_norm_weight: R.Tensor((64,), dtype="float32"), mask_decoder_upscale_layer_norm_bias: R.Tensor((64,), dtype="float32"), mask_decoder_output_hypernetworks_mlps_0_proj_in_weight: R.Tensor((256, 256), dtype="float32"), mask_decoder_output_hypernetworks_mlps_0_proj_in_bias: R.Tensor((256,), dtype="float32"), mask_decoder_output_hypernetworks_mlps_0_proj_out_weight: R.Tensor((32, 256), dtype="float32"), mask_decoder_output_hypernetworks_mlps_0_proj_out_bias: R.Tensor((32,), dtype="float32"), mask_decoder_output_hypernetworks_mlps_0_layers_0_weight: R.Tensor((256, 256), dtype="float32"), mask_decoder_output_hypernetworks_mlps_0_layers_0_bias: R.Tensor((256,), dtype="float32"), mask_decoder_output_hypernetworks_mlps_1_proj_in_weight: R.Tensor((256, 256), dtype="float32"), mask_decoder_output_hypernetworks_mlps_1_proj_in_bias: R.Tensor((256,), dtype="float32"), mask_decoder_output_hypernetworks_mlps_1_proj_out_weight: R.Tensor((32, 256), dtype="float32"), mask_decoder_output_hypernetworks_mlps_1_proj_out_bias: R.Tensor((32,), dtype="float32"), mask_decoder_output_hypernetworks_mlps_1_layers_0_weight: R.Tensor((256, 256), dtype="float32"), mask_decoder_output_hypernetworks_mlps_1_layers_0_bias: R.Tensor((256,), dtype="float32"), mask_decoder_output_hypernetworks_mlps_2_proj_in_weight: R.Tensor((256, 256), dtype="float32"), mask_decoder_output_hypernetworks_mlps_2_proj_in_bias: R.Tensor((256,), dtype="float32"), mask_decoder_output_hypernetworks_mlps_2_proj_out_weight: R.Tensor((32, 256), dtype="float32"), mask_decoder_output_hypernetworks_mlps_2_proj_out_bias: R.Tensor((32,), dtype="float32"), mask_decoder_output_hypernetworks_mlps_2_layers_0_weight: R.Tensor((256, 256), dtype="float32"), mask_decoder_output_hypernetworks_mlps_2_layers_0_bias: R.Tensor((256,), dtype="float32"), mask_decoder_output_hypernetworks_mlps_3_proj_in_weight: R.Tensor((256, 256), dtype="float32"), mask_decoder_output_hypernetworks_mlps_3_proj_in_bias: R.Tensor((256,), dtype="float32"), mask_decoder_output_hypernetworks_mlps_3_proj_out_weight: R.Tensor((32, 256), dtype="float32"), mask_decoder_output_hypernetworks_mlps_3_proj_out_bias: R.Tensor((32,), dtype="float32"), mask_decoder_output_hypernetworks_mlps_3_layers_0_weight: R.Tensor((256, 256), dtype="float32"), mask_decoder_output_hypernetworks_mlps_3_layers_0_bias: R.Tensor((256,), dtype="float32"), mask_decoder_iou_prediction_head_proj_in_weight: R.Tensor((256, 256), dtype="float32"), mask_decoder_iou_prediction_head_proj_in_bias: R.Tensor((256,), dtype="float32"), mask_decoder_iou_prediction_head_proj_out_weight: R.Tensor((4, 256), dtype="float32"), mask_decoder_iou_prediction_head_proj_out_bias: R.Tensor((4,), dtype="float32"), mask_decoder_iou_prediction_head_layers_0_weight: R.Tensor((256, 256), dtype="float32"), mask_decoder_iou_prediction_head_layers_0_bias: R.Tensor((256,), dtype="float32")) -> R.Tuple(R.Tuple(R.Tensor((1, 1, 3), dtype="float32"), R.Tensor((1, 1, 3, 256, 256), dtype="float32")), R.Tuple(R.Object)):
        R.func_attr({"global_symbol": "main", "num_input": 3})
        with R.dataflow():
            matmul73: R.Tensor((64, 64, 128), dtype="float32") = R.matmul(metadata["relax.expr.Constant"][5], prompt_encoder_shared_embedding_positional_embedding, out_dtype="void")
            mul64: R.Tensor((64, 64, 128), dtype="float32") = R.multiply(R.const(6.2831854820251465, "float32"), matmul73)
            sin1: R.Tensor((64, 64, 128), dtype="float32") = R.sin(mul64)
            cos1: R.Tensor((64, 64, 128), dtype="float32") = R.cos(mul64)
            concat5: R.Tensor((64, 64, 256), dtype="float32") = R.concat((sin1, cos1), axis=2)
            permute_dims150: R.Tensor((256, 64, 64), dtype="float32") = R.permute_dims(concat5, axes=[2, 0, 1])
            unsqueeze: R.Tensor((1, 256, 64, 64), dtype="float32") = R.expand_dims(permute_dims150, axis=[0])
            repeat: R.Tensor((1, 256, 64, 64), dtype="float32") = R.repeat(unsqueeze, repeats=1, axis=0)
            lv: R.Tensor((1, 1024, 1024, 3), dtype="float32") = R.permute_dims(pixel_values, axes=[0, 2, 3, 1])
            lv1: R.Tensor((768, 16, 16, 3), dtype="float32") = R.permute_dims(vision_encoder_patch_embed_projection_weight, axes=[0, 2, 3, 1])
            lv5: R.Tensor((1, 64, 64, 768), dtype="float32") = R.nn.conv2d(lv, lv1, strides=[16, 16], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NHWC", kernel_layout="OHWI", out_layout="NHWC", out_dtype="void")
            lv6: R.Tensor((1, 768, 1, 1), dtype="float32") = R.reshape(vision_encoder_patch_embed_projection_bias, R.shape([1, 768, 1, 1]))
            lv2: R.Tensor((1, 1, 1, 768), dtype="float32") = R.permute_dims(lv6, axes=[0, 2, 3, 1])
            conv2d3: R.Tensor((1, 64, 64, 768), dtype="float32") = R.add(lv5, lv2)
            permute_dims151: R.Tensor((1, 64, 64, 768), dtype="float32") = R.permute_dims(conv2d3, axes=[0, 1, 2, 3])
            add130: R.Tensor((1, 64, 64, 768), dtype="float32") = R.add(permute_dims151, vision_encoder_pos_embed)
            layer_norm24: R.Tensor((1, 64, 64, 768), dtype="float32") = R.nn.layer_norm(add130, vision_encoder_layers_0_layer_norm1_weight, vision_encoder_layers_0_layer_norm1_bias, axes=[3], epsilon=9.9999999999999995e-07, center=True, scale=True)
            pad8: R.Tensor((1, 70, 70, 768), dtype="float32") = R.nn.pad(layer_norm24, [0, 0, 0, 6, 0, 6, 0, 0], R.const(0, "int32"), pad_mode="constant")
            reshape165: R.Tensor((1, 5, 14, 5, 14, 768), dtype="float32") = R.reshape(pad8, R.shape([1, 5, 14, 5, 14, 768]))
            permute_dims152: R.Tensor((1, 5, 5, 14, 14, 768), dtype="float32") = R.permute_dims(reshape165, axes=[0, 1, 3, 2, 4, 5])
            reshape166: R.Tensor((25, 14, 14, 768), dtype="float32") = R.reshape(permute_dims152, R.shape([25, 14, 14, 768]))
            permute_dims153: R.Tensor((768, 2304), dtype="float32") = R.permute_dims(vision_encoder_layers_0_attn_qkv_weight, axes=[1, 0])
            matmul74: R.Tensor((25, 14, 14, 2304), dtype="float32") = R.matmul(reshape166, permute_dims153, out_dtype="void")
            add131: R.Tensor((25, 14, 14, 2304), dtype="float32") = R.add(matmul74, vision_encoder_layers_0_attn_qkv_bias)
            reshape167: R.Tensor((25, 196, 3, 12, 64), dtype="float32") = R.reshape(add131, R.shape([25, 196, 3, 12, 64]))
            permute_dims154: R.Tensor((3, 25, 12, 196, 64), dtype="float32") = R.permute_dims(reshape167, axes=[2, 0, 3, 1, 4])
            reshape168: R.Tensor((3, 300, 196, 64), dtype="float32") = R.reshape(permute_dims154, R.shape([3, 300, 196, 64]))
            split12: R.Tuple(R.Tensor((1, 300, 196, 64), dtype="float32"), R.Tensor((1, 300, 196, 64), dtype="float32"), R.Tensor((1, 300, 196, 64), dtype="float32")) = R.split(reshape168, indices_or_sections=3, axis=0)
            split_012: R.Tensor((1, 300, 196, 64), dtype="float32") = split12[0]
            split_112: R.Tensor((1, 300, 196, 64), dtype="float32") = split12[1]
            split_212: R.Tensor((1, 300, 196, 64), dtype="float32") = split12[2]
            unbind36: R.Tensor((300, 196, 64), dtype="float32") = R.squeeze(split_012, axis=[0])
            unbind37: R.Tensor((300, 196, 64), dtype="float32") = R.squeeze(split_112, axis=[0])
            unbind38: R.Tensor((300, 196, 64), dtype="float32") = R.squeeze(split_212, axis=[0])
            mul65: R.Tensor((300, 196, 64), dtype="float32") = R.multiply(unbind36, R.const(0.125, "float32"))
            permute_dims155: R.Tensor((300, 64, 196), dtype="float32") = R.permute_dims(unbind37, axes=[0, 2, 1])
            matmul75: R.Tensor((300, 196, 196), dtype="float32") = R.matmul(mul65, permute_dims155, out_dtype="void")
            reshape169: R.Tensor((1, 27, 64), dtype="float32") = R.reshape(vision_encoder_layers_0_attn_rel_pos_h, R.shape([1, 27, 64]))
            permute_dims156: R.Tensor((1, 64, 27), dtype="float32") = R.permute_dims(reshape169, axes=[0, 2, 1])
            reshape170: R.Tensor((64, 27), dtype="float32") = R.reshape(permute_dims156, R.shape([64, 27]))
            permute_dims157: R.Tensor((27, 64), dtype="float32") = R.permute_dims(reshape170, axes=[1, 0])
            arange48: R.Tensor((14,), dtype="int64") = R.arange(R.prim_value(0), R.prim_value(14), R.prim_value(1), dtype="int64")
            expand_dims89: R.Tensor((1, 14), dtype="int64") = R.expand_dims(arange48, axis=[0])
            mul66: R.Tensor((1, 14), dtype="int64") = R.multiply(expand_dims89, R.const(1, "int64"))
            expand_dims90: R.Tensor((14, 1), dtype="int64") = R.expand_dims(arange48, axis=[1])
            mul67: R.Tensor((14, 1), dtype="int64") = R.multiply(expand_dims90, R.const(1, "int64"))
            subtract30: R.Tensor((14, 14), dtype="int64") = R.subtract(mul66, mul67)
            add132: R.Tensor((14, 14), dtype="int64") = R.add(subtract30, R.const(13, "int64"))
            take24: R.Tensor((14, 14, 64), dtype="float32") = R.take(permute_dims157, add132, axis=0)
            reshape171: R.Tensor((1, 27, 64), dtype="float32") = R.reshape(vision_encoder_layers_0_attn_rel_pos_w, R.shape([1, 27, 64]))
            permute_dims158: R.Tensor((1, 64, 27), dtype="float32") = R.permute_dims(reshape171, axes=[0, 2, 1])
            reshape172: R.Tensor((64, 27), dtype="float32") = R.reshape(permute_dims158, R.shape([64, 27]))
            permute_dims159: R.Tensor((27, 64), dtype="float32") = R.permute_dims(reshape172, axes=[1, 0])
            take25: R.Tensor((14, 14, 64), dtype="float32") = R.take(permute_dims159, add132, axis=0)
            reshape173: R.Tensor((300, 14, 14, 64), dtype="float32") = R.reshape(unbind36, R.shape([300, 14, 14, 64]))
            einsum24: R.Tensor((300, 14, 14, 14), dtype="float32") = R.einsum((reshape173, take24), subscripts="bhwc,hkc->bhwk")
            einsum25: R.Tensor((300, 14, 14, 14), dtype="float32") = R.einsum((reshape173, take25), subscripts="bhwc,wkc->bhwk")
            reshape174: R.Tensor((300, 14, 14, 14, 14), dtype="float32") = R.reshape(matmul75, R.shape([300, 14, 14, 14, 14]))
            expand_dims93: R.Tensor((300, 14, 14, 1, 14), dtype="float32") = R.expand_dims(einsum24, axis=[3])
            expand_dims94: R.Tensor((300, 14, 1, 14, 14), dtype="float32") = R.expand_dims(einsum25, axis=[2])
            add134: R.Tensor((300, 14, 14, 14, 14), dtype="float32") = R.add(reshape174, expand_dims93)
            add135: R.Tensor((300, 14, 14, 14, 14), dtype="float32") = R.add(add134, expand_dims94)
            reshape175: R.Tensor((300, 196, 196), dtype="float32") = R.reshape(add135, R.shape([300, 196, 196]))
            softmax12: R.Tensor((300, 196, 196), dtype="float32") = R.nn.softmax(reshape175, axis=2)
            matmul76: R.Tensor((300, 196, 64), dtype="float32") = R.matmul(softmax12, unbind38, out_dtype="void")
            reshape176: R.Tensor((25, 12, 14, 14, 64), dtype="float32") = R.reshape(matmul76, R.shape([25, 12, 14, 14, 64]))
            permute_dims160: R.Tensor((25, 14, 14, 12, 64), dtype="float32") = R.permute_dims(reshape176, axes=[0, 2, 3, 1, 4])
            reshape177: R.Tensor((25, 14, 14, 768), dtype="float32") = R.reshape(permute_dims160, R.shape([25, 14, 14, 768]))
            permute_dims161: R.Tensor((768, 768), dtype="float32") = R.permute_dims(vision_encoder_layers_0_attn_proj_weight, axes=[1, 0])
            matmul77: R.Tensor((25, 14, 14, 768), dtype="float32") = R.matmul(reshape177, permute_dims161, out_dtype="void")
            add136: R.Tensor((25, 14, 14, 768), dtype="float32") = R.add(matmul77, vision_encoder_layers_0_attn_proj_bias)
            reshape178: R.Tensor((1, 5, 5, 14, 14, 768), dtype="float32") = R.reshape(add136, R.shape([1, 5, 5, 14, 14, 768]))
            permute_dims162: R.Tensor((1, 5, 14, 5, 14, 768), dtype="float32") = R.permute_dims(reshape178, axes=[0, 1, 3, 2, 4, 5])
            reshape179: R.Tensor((1, 70, 70, 768), dtype="float32") = R.reshape(permute_dims162, R.shape([1, 70, 70, 768]))
            strided_slice10: R.Tensor((1, 64, 64, 768), dtype="float32") = R.strided_slice(reshape179, axes=[1, 2], begin=[0, 0], end=[64, 64], strides=None, assume_inbound=False)
            add137: R.Tensor((1, 64, 64, 768), dtype="float32") = R.add(add130, strided_slice10)
            layer_norm25: R.Tensor((1, 64, 64, 768), dtype="float32") = R.nn.layer_norm(add137, vision_encoder_layers_0_layer_norm2_weight, vision_encoder_layers_0_layer_norm2_bias, axes=[3], epsilon=9.9999999999999995e-07, center=True, scale=True)
            permute_dims163: R.Tensor((768, 3072), dtype="float32") = R.permute_dims(vision_encoder_layers_0_mlp_lin1_weight, axes=[1, 0])
            matmul78: R.Tensor((1, 64, 64, 3072), dtype="float32") = R.matmul(layer_norm25, permute_dims163, out_dtype="void")
            add138: R.Tensor((1, 64, 64, 3072), dtype="float32") = R.add(matmul78, vision_encoder_layers_0_mlp_lin1_bias)
            gelu12: R.Tensor((1, 64, 64, 3072), dtype="float32") = R.nn.gelu(add138)
            permute_dims164: R.Tensor((3072, 768), dtype="float32") = R.permute_dims(vision_encoder_layers_0_mlp_lin2_weight, axes=[1, 0])
            matmul79: R.Tensor((1, 64, 64, 768), dtype="float32") = R.matmul(gelu12, permute_dims164, out_dtype="void")
            add139: R.Tensor((1, 64, 64, 768), dtype="float32") = R.add(matmul79, vision_encoder_layers_0_mlp_lin2_bias)
            add140: R.Tensor((1, 64, 64, 768), dtype="float32") = R.add(add137, add139)
            layer_norm26: R.Tensor((1, 64, 64, 768), dtype="float32") = R.nn.layer_norm(add140, vision_encoder_layers_1_layer_norm1_weight, vision_encoder_layers_1_layer_norm1_bias, axes=[3], epsilon=9.9999999999999995e-07, center=True, scale=True)
            pad9: R.Tensor((1, 70, 70, 768), dtype="float32") = R.nn.pad(layer_norm26, [0, 0, 0, 6, 0, 6, 0, 0], R.const(0, "int32"), pad_mode="constant")
            reshape180: R.Tensor((1, 5, 14, 5, 14, 768), dtype="float32") = R.reshape(pad9, R.shape([1, 5, 14, 5, 14, 768]))
            permute_dims165: R.Tensor((1, 5, 5, 14, 14, 768), dtype="float32") = R.permute_dims(reshape180, axes=[0, 1, 3, 2, 4, 5])
            reshape181: R.Tensor((25, 14, 14, 768), dtype="float32") = R.reshape(permute_dims165, R.shape([25, 14, 14, 768]))
            permute_dims166: R.Tensor((768, 2304), dtype="float32") = R.permute_dims(vision_encoder_layers_1_attn_qkv_weight, axes=[1, 0])
            matmul80: R.Tensor((25, 14, 14, 2304), dtype="float32") = R.matmul(reshape181, permute_dims166, out_dtype="void")
            add141: R.Tensor((25, 14, 14, 2304), dtype="float32") = R.add(matmul80, vision_encoder_layers_1_attn_qkv_bias)
            reshape182: R.Tensor((25, 196, 3, 12, 64), dtype="float32") = R.reshape(add141, R.shape([25, 196, 3, 12, 64]))
            permute_dims167: R.Tensor((3, 25, 12, 196, 64), dtype="float32") = R.permute_dims(reshape182, axes=[2, 0, 3, 1, 4])
            reshape183: R.Tensor((3, 300, 196, 64), dtype="float32") = R.reshape(permute_dims167, R.shape([3, 300, 196, 64]))
            split13: R.Tuple(R.Tensor((1, 300, 196, 64), dtype="float32"), R.Tensor((1, 300, 196, 64), dtype="float32"), R.Tensor((1, 300, 196, 64), dtype="float32")) = R.split(reshape183, indices_or_sections=3, axis=0)
            split_013: R.Tensor((1, 300, 196, 64), dtype="float32") = split13[0]
            split_113: R.Tensor((1, 300, 196, 64), dtype="float32") = split13[1]
            split_213: R.Tensor((1, 300, 196, 64), dtype="float32") = split13[2]
            unbind39: R.Tensor((300, 196, 64), dtype="float32") = R.squeeze(split_013, axis=[0])
            unbind40: R.Tensor((300, 196, 64), dtype="float32") = R.squeeze(split_113, axis=[0])
            unbind41: R.Tensor((300, 196, 64), dtype="float32") = R.squeeze(split_213, axis=[0])
            mul70: R.Tensor((300, 196, 64), dtype="float32") = R.multiply(unbind39, R.const(0.125, "float32"))
            permute_dims168: R.Tensor((300, 64, 196), dtype="float32") = R.permute_dims(unbind40, axes=[0, 2, 1])
            matmul81: R.Tensor((300, 196, 196), dtype="float32") = R.matmul(mul70, permute_dims168, out_dtype="void")
            reshape184: R.Tensor((1, 27, 64), dtype="float32") = R.reshape(vision_encoder_layers_1_attn_rel_pos_h, R.shape([1, 27, 64]))
            permute_dims169: R.Tensor((1, 64, 27), dtype="float32") = R.permute_dims(reshape184, axes=[0, 2, 1])
            reshape185: R.Tensor((64, 27), dtype="float32") = R.reshape(permute_dims169, R.shape([64, 27]))
            permute_dims170: R.Tensor((27, 64), dtype="float32") = R.permute_dims(reshape185, axes=[1, 0])
            take26: R.Tensor((14, 14, 64), dtype="float32") = R.take(permute_dims170, add132, axis=0)
            reshape186: R.Tensor((1, 27, 64), dtype="float32") = R.reshape(vision_encoder_layers_1_attn_rel_pos_w, R.shape([1, 27, 64]))
            permute_dims171: R.Tensor((1, 64, 27), dtype="float32") = R.permute_dims(reshape186, axes=[0, 2, 1])
            reshape187: R.Tensor((64, 27), dtype="float32") = R.reshape(permute_dims171, R.shape([64, 27]))
            permute_dims172: R.Tensor((27, 64), dtype="float32") = R.permute_dims(reshape187, axes=[1, 0])
            take27: R.Tensor((14, 14, 64), dtype="float32") = R.take(permute_dims172, add132, axis=0)
            reshape188: R.Tensor((300, 14, 14, 64), dtype="float32") = R.reshape(unbind39, R.shape([300, 14, 14, 64]))
            einsum26: R.Tensor((300, 14, 14, 14), dtype="float32") = R.einsum((reshape188, take26), subscripts="bhwc,hkc->bhwk")
            einsum27: R.Tensor((300, 14, 14, 14), dtype="float32") = R.einsum((reshape188, take27), subscripts="bhwc,wkc->bhwk")
            reshape189: R.Tensor((300, 14, 14, 14, 14), dtype="float32") = R.reshape(matmul81, R.shape([300, 14, 14, 14, 14]))
            expand_dims99: R.Tensor((300, 14, 14, 1, 14), dtype="float32") = R.expand_dims(einsum26, axis=[3])
            expand_dims100: R.Tensor((300, 14, 1, 14, 14), dtype="float32") = R.expand_dims(einsum27, axis=[2])
            add144: R.Tensor((300, 14, 14, 14, 14), dtype="float32") = R.add(reshape189, expand_dims99)
            add145: R.Tensor((300, 14, 14, 14, 14), dtype="float32") = R.add(add144, expand_dims100)
            reshape190: R.Tensor((300, 196, 196), dtype="float32") = R.reshape(add145, R.shape([300, 196, 196]))
            softmax13: R.Tensor((300, 196, 196), dtype="float32") = R.nn.softmax(reshape190, axis=2)
            matmul82: R.Tensor((300, 196, 64), dtype="float32") = R.matmul(softmax13, unbind41, out_dtype="void")
            reshape191: R.Tensor((25, 12, 14, 14, 64), dtype="float32") = R.reshape(matmul82, R.shape([25, 12, 14, 14, 64]))
            permute_dims173: R.Tensor((25, 14, 14, 12, 64), dtype="float32") = R.permute_dims(reshape191, axes=[0, 2, 3, 1, 4])
            reshape192: R.Tensor((25, 14, 14, 768), dtype="float32") = R.reshape(permute_dims173, R.shape([25, 14, 14, 768]))
            permute_dims174: R.Tensor((768, 768), dtype="float32") = R.permute_dims(vision_encoder_layers_1_attn_proj_weight, axes=[1, 0])
            matmul83: R.Tensor((25, 14, 14, 768), dtype="float32") = R.matmul(reshape192, permute_dims174, out_dtype="void")
            add146: R.Tensor((25, 14, 14, 768), dtype="float32") = R.add(matmul83, vision_encoder_layers_1_attn_proj_bias)
            reshape193: R.Tensor((1, 5, 5, 14, 14, 768), dtype="float32") = R.reshape(add146, R.shape([1, 5, 5, 14, 14, 768]))
            permute_dims175: R.Tensor((1, 5, 14, 5, 14, 768), dtype="float32") = R.permute_dims(reshape193, axes=[0, 1, 3, 2, 4, 5])
            reshape194: R.Tensor((1, 70, 70, 768), dtype="float32") = R.reshape(permute_dims175, R.shape([1, 70, 70, 768]))
            strided_slice11: R.Tensor((1, 64, 64, 768), dtype="float32") = R.strided_slice(reshape194, axes=[1, 2], begin=[0, 0], end=[64, 64], strides=None, assume_inbound=False)
            add147: R.Tensor((1, 64, 64, 768), dtype="float32") = R.add(add140, strided_slice11)
            layer_norm27: R.Tensor((1, 64, 64, 768), dtype="float32") = R.nn.layer_norm(add147, vision_encoder_layers_1_layer_norm2_weight, vision_encoder_layers_1_layer_norm2_bias, axes=[3], epsilon=9.9999999999999995e-07, center=True, scale=True)
            permute_dims176: R.Tensor((768, 3072), dtype="float32") = R.permute_dims(vision_encoder_layers_1_mlp_lin1_weight, axes=[1, 0])
            matmul84: R.Tensor((1, 64, 64, 3072), dtype="float32") = R.matmul(layer_norm27, permute_dims176, out_dtype="void")
            add148: R.Tensor((1, 64, 64, 3072), dtype="float32") = R.add(matmul84, vision_encoder_layers_1_mlp_lin1_bias)
            gelu13: R.Tensor((1, 64, 64, 3072), dtype="float32") = R.nn.gelu(add148)
            permute_dims177: R.Tensor((3072, 768), dtype="float32") = R.permute_dims(vision_encoder_layers_1_mlp_lin2_weight, axes=[1, 0])
            matmul85: R.Tensor((1, 64, 64, 768), dtype="float32") = R.matmul(gelu13, permute_dims177, out_dtype="void")
            add149: R.Tensor((1, 64, 64, 768), dtype="float32") = R.add(matmul85, vision_encoder_layers_1_mlp_lin2_bias)
            add150: R.Tensor((1, 64, 64, 768), dtype="float32") = R.add(add147, add149)
            layer_norm28: R.Tensor((1, 64, 64, 768), dtype="float32") = R.nn.layer_norm(add150, vision_encoder_layers_2_layer_norm1_weight, vision_encoder_layers_2_layer_norm1_bias, axes=[3], epsilon=9.9999999999999995e-07, center=True, scale=True)
            permute_dims178: R.Tensor((768, 2304), dtype="float32") = R.permute_dims(vision_encoder_layers_2_attn_qkv_weight, axes=[1, 0])
            matmul86: R.Tensor((1, 64, 64, 2304), dtype="float32") = R.matmul(layer_norm28, permute_dims178, out_dtype="void")
            add151: R.Tensor((1, 64, 64, 2304), dtype="float32") = R.add(matmul86, vision_encoder_layers_2_attn_qkv_bias)
            reshape195: R.Tensor((1, 4096, 3, 12, 64), dtype="float32") = R.reshape(add151, R.shape([1, 4096, 3, 12, 64]))
            permute_dims179: R.Tensor((3, 1, 12, 4096, 64), dtype="float32") = R.permute_dims(reshape195, axes=[2, 0, 3, 1, 4])
            reshape196: R.Tensor((3, 12, 4096, 64), dtype="float32") = R.reshape(permute_dims179, R.shape([3, 12, 4096, 64]))
            split14: R.Tuple(R.Tensor((1, 12, 4096, 64), dtype="float32"), R.Tensor((1, 12, 4096, 64), dtype="float32"), R.Tensor((1, 12, 4096, 64), dtype="float32")) = R.split(reshape196, indices_or_sections=3, axis=0)
            split_014: R.Tensor((1, 12, 4096, 64), dtype="float32") = split14[0]
            split_114: R.Tensor((1, 12, 4096, 64), dtype="float32") = split14[1]
            split_214: R.Tensor((1, 12, 4096, 64), dtype="float32") = split14[2]
            unbind42: R.Tensor((12, 4096, 64), dtype="float32") = R.squeeze(split_014, axis=[0])
            unbind43: R.Tensor((12, 4096, 64), dtype="float32") = R.squeeze(split_114, axis=[0])
            unbind44: R.Tensor((12, 4096, 64), dtype="float32") = R.squeeze(split_214, axis=[0])
            mul75: R.Tensor((12, 4096, 64), dtype="float32") = R.multiply(unbind42, R.const(0.125, "float32"))
            permute_dims180: R.Tensor((12, 64, 4096), dtype="float32") = R.permute_dims(unbind43, axes=[0, 2, 1])
            matmul87: R.Tensor((12, 4096, 4096), dtype="float32") = R.matmul(mul75, permute_dims180, out_dtype="void")
            reshape197: R.Tensor((1, 127, 64), dtype="float32") = R.reshape(vision_encoder_layers_2_attn_rel_pos_h, R.shape([1, 127, 64]))
            permute_dims181: R.Tensor((1, 64, 127), dtype="float32") = R.permute_dims(reshape197, axes=[0, 2, 1])
            reshape198: R.Tensor((64, 127), dtype="float32") = R.reshape(permute_dims181, R.shape([64, 127]))
            permute_dims182: R.Tensor((127, 64), dtype="float32") = R.permute_dims(reshape198, axes=[1, 0])
            arange56: R.Tensor((64,), dtype="int64") = R.arange(R.prim_value(0), R.prim_value(64), R.prim_value(1), dtype="int64")
            expand_dims101: R.Tensor((1, 64), dtype="int64") = R.expand_dims(arange56, axis=[0])
            mul76: R.Tensor((1, 64), dtype="int64") = R.multiply(expand_dims101, R.const(1, "int64"))
            expand_dims102: R.Tensor((64, 1), dtype="int64") = R.expand_dims(arange56, axis=[1])
            mul77: R.Tensor((64, 1), dtype="int64") = R.multiply(expand_dims102, R.const(1, "int64"))
            subtract34: R.Tensor((64, 64), dtype="int64") = R.subtract(mul76, mul77)
            add152: R.Tensor((64, 64), dtype="int64") = R.add(subtract34, R.const(63, "int64"))
            take28: R.Tensor((64, 64, 64), dtype="float32") = R.take(permute_dims182, add152, axis=0)
            reshape199: R.Tensor((1, 127, 64), dtype="float32") = R.reshape(vision_encoder_layers_2_attn_rel_pos_w, R.shape([1, 127, 64]))
            permute_dims183: R.Tensor((1, 64, 127), dtype="float32") = R.permute_dims(reshape199, axes=[0, 2, 1])
            reshape200: R.Tensor((64, 127), dtype="float32") = R.reshape(permute_dims183, R.shape([64, 127]))
            permute_dims184: R.Tensor((127, 64), dtype="float32") = R.permute_dims(reshape200, axes=[1, 0])
            take29: R.Tensor((64, 64, 64), dtype="float32") = R.take(permute_dims184, add152, axis=0)
            reshape201: R.Tensor((12, 64, 64, 64), dtype="float32") = R.reshape(unbind42, R.shape([12, 64, 64, 64]))
            einsum28: R.Tensor((12, 64, 64, 64), dtype="float32") = R.einsum((reshape201, take28), subscripts="bhwc,hkc->bhwk")
            einsum29: R.Tensor((12, 64, 64, 64), dtype="float32") = R.einsum((reshape201, take29), subscripts="bhwc,wkc->bhwk")
            reshape202: R.Tensor((12, 64, 64, 64, 64), dtype="float32") = R.reshape(matmul87, R.shape([12, 64, 64, 64, 64]))
            expand_dims105: R.Tensor((12, 64, 64, 1, 64), dtype="float32") = R.expand_dims(einsum28, axis=[3])
            expand_dims106: R.Tensor((12, 64, 1, 64, 64), dtype="float32") = R.expand_dims(einsum29, axis=[2])
            add154: R.Tensor((12, 64, 64, 64, 64), dtype="float32") = R.add(reshape202, expand_dims105)
            add155: R.Tensor((12, 64, 64, 64, 64), dtype="float32") = R.add(add154, expand_dims106)
            reshape203: R.Tensor((12, 4096, 4096), dtype="float32") = R.reshape(add155, R.shape([12, 4096, 4096]))
            softmax14: R.Tensor((12, 4096, 4096), dtype="float32") = R.nn.softmax(reshape203, axis=2)
            matmul88: R.Tensor((12, 4096, 64), dtype="float32") = R.matmul(softmax14, unbind44, out_dtype="void")
            reshape204: R.Tensor((1, 12, 64, 64, 64), dtype="float32") = R.reshape(matmul88, R.shape([1, 12, 64, 64, 64]))
            permute_dims185: R.Tensor((1, 64, 64, 12, 64), dtype="float32") = R.permute_dims(reshape204, axes=[0, 2, 3, 1, 4])
            reshape205: R.Tensor((1, 64, 64, 768), dtype="float32") = R.reshape(permute_dims185, R.shape([1, 64, 64, 768]))
            permute_dims186: R.Tensor((768, 768), dtype="float32") = R.permute_dims(vision_encoder_layers_2_attn_proj_weight, axes=[1, 0])
            matmul89: R.Tensor((1, 64, 64, 768), dtype="float32") = R.matmul(reshape205, permute_dims186, out_dtype="void")
            add156: R.Tensor((1, 64, 64, 768), dtype="float32") = R.add(matmul89, vision_encoder_layers_2_attn_proj_bias)
            add157: R.Tensor((1, 64, 64, 768), dtype="float32") = R.add(add150, add156)
            layer_norm29: R.Tensor((1, 64, 64, 768), dtype="float32") = R.nn.layer_norm(add157, vision_encoder_layers_2_layer_norm2_weight, vision_encoder_layers_2_layer_norm2_bias, axes=[3], epsilon=9.9999999999999995e-07, center=True, scale=True)
            permute_dims187: R.Tensor((768, 3072), dtype="float32") = R.permute_dims(vision_encoder_layers_2_mlp_lin1_weight, axes=[1, 0])
            matmul90: R.Tensor((1, 64, 64, 3072), dtype="float32") = R.matmul(layer_norm29, permute_dims187, out_dtype="void")
            add158: R.Tensor((1, 64, 64, 3072), dtype="float32") = R.add(matmul90, vision_encoder_layers_2_mlp_lin1_bias)
            gelu14: R.Tensor((1, 64, 64, 3072), dtype="float32") = R.nn.gelu(add158)
            permute_dims188: R.Tensor((3072, 768), dtype="float32") = R.permute_dims(vision_encoder_layers_2_mlp_lin2_weight, axes=[1, 0])
            matmul91: R.Tensor((1, 64, 64, 768), dtype="float32") = R.matmul(gelu14, permute_dims188, out_dtype="void")
            add159: R.Tensor((1, 64, 64, 768), dtype="float32") = R.add(matmul91, vision_encoder_layers_2_mlp_lin2_bias)
            add160: R.Tensor((1, 64, 64, 768), dtype="float32") = R.add(add157, add159)
            layer_norm30: R.Tensor((1, 64, 64, 768), dtype="float32") = R.nn.layer_norm(add160, vision_encoder_layers_3_layer_norm1_weight, vision_encoder_layers_3_layer_norm1_bias, axes=[3], epsilon=9.9999999999999995e-07, center=True, scale=True)
            pad10: R.Tensor((1, 70, 70, 768), dtype="float32") = R.nn.pad(layer_norm30, [0, 0, 0, 6, 0, 6, 0, 0], R.const(0, "int32"), pad_mode="constant")
            reshape206: R.Tensor((1, 5, 14, 5, 14, 768), dtype="float32") = R.reshape(pad10, R.shape([1, 5, 14, 5, 14, 768]))
            permute_dims189: R.Tensor((1, 5, 5, 14, 14, 768), dtype="float32") = R.permute_dims(reshape206, axes=[0, 1, 3, 2, 4, 5])
            reshape207: R.Tensor((25, 14, 14, 768), dtype="float32") = R.reshape(permute_dims189, R.shape([25, 14, 14, 768]))
            permute_dims190: R.Tensor((768, 2304), dtype="float32") = R.permute_dims(vision_encoder_layers_3_attn_qkv_weight, axes=[1, 0])
            matmul92: R.Tensor((25, 14, 14, 2304), dtype="float32") = R.matmul(reshape207, permute_dims190, out_dtype="void")
            add161: R.Tensor((25, 14, 14, 2304), dtype="float32") = R.add(matmul92, vision_encoder_layers_3_attn_qkv_bias)
            reshape208: R.Tensor((25, 196, 3, 12, 64), dtype="float32") = R.reshape(add161, R.shape([25, 196, 3, 12, 64]))
            permute_dims191: R.Tensor((3, 25, 12, 196, 64), dtype="float32") = R.permute_dims(reshape208, axes=[2, 0, 3, 1, 4])
            reshape209: R.Tensor((3, 300, 196, 64), dtype="float32") = R.reshape(permute_dims191, R.shape([3, 300, 196, 64]))
            split15: R.Tuple(R.Tensor((1, 300, 196, 64), dtype="float32"), R.Tensor((1, 300, 196, 64), dtype="float32"), R.Tensor((1, 300, 196, 64), dtype="float32")) = R.split(reshape209, indices_or_sections=3, axis=0)
            split_015: R.Tensor((1, 300, 196, 64), dtype="float32") = split15[0]
            split_115: R.Tensor((1, 300, 196, 64), dtype="float32") = split15[1]
            split_215: R.Tensor((1, 300, 196, 64), dtype="float32") = split15[2]
            unbind45: R.Tensor((300, 196, 64), dtype="float32") = R.squeeze(split_015, axis=[0])
            unbind46: R.Tensor((300, 196, 64), dtype="float32") = R.squeeze(split_115, axis=[0])
            unbind47: R.Tensor((300, 196, 64), dtype="float32") = R.squeeze(split_215, axis=[0])
            mul80: R.Tensor((300, 196, 64), dtype="float32") = R.multiply(unbind45, R.const(0.125, "float32"))
            permute_dims192: R.Tensor((300, 64, 196), dtype="float32") = R.permute_dims(unbind46, axes=[0, 2, 1])
            matmul93: R.Tensor((300, 196, 196), dtype="float32") = R.matmul(mul80, permute_dims192, out_dtype="void")
            reshape210: R.Tensor((1, 27, 64), dtype="float32") = R.reshape(vision_encoder_layers_3_attn_rel_pos_h, R.shape([1, 27, 64]))
            permute_dims193: R.Tensor((1, 64, 27), dtype="float32") = R.permute_dims(reshape210, axes=[0, 2, 1])
            reshape211: R.Tensor((64, 27), dtype="float32") = R.reshape(permute_dims193, R.shape([64, 27]))
            permute_dims194: R.Tensor((27, 64), dtype="float32") = R.permute_dims(reshape211, axes=[1, 0])
            take30: R.Tensor((14, 14, 64), dtype="float32") = R.take(permute_dims194, add132, axis=0)
            reshape212: R.Tensor((1, 27, 64), dtype="float32") = R.reshape(vision_encoder_layers_3_attn_rel_pos_w, R.shape([1, 27, 64]))
            permute_dims195: R.Tensor((1, 64, 27), dtype="float32") = R.permute_dims(reshape212, axes=[0, 2, 1])
            reshape213: R.Tensor((64, 27), dtype="float32") = R.reshape(permute_dims195, R.shape([64, 27]))
            permute_dims196: R.Tensor((27, 64), dtype="float32") = R.permute_dims(reshape213, axes=[1, 0])
            take31: R.Tensor((14, 14, 64), dtype="float32") = R.take(permute_dims196, add132, axis=0)
            reshape214: R.Tensor((300, 14, 14, 64), dtype="float32") = R.reshape(unbind45, R.shape([300, 14, 14, 64]))
            einsum30: R.Tensor((300, 14, 14, 14), dtype="float32") = R.einsum((reshape214, take30), subscripts="bhwc,hkc->bhwk")
            einsum31: R.Tensor((300, 14, 14, 14), dtype="float32") = R.einsum((reshape214, take31), subscripts="bhwc,wkc->bhwk")
            reshape215: R.Tensor((300, 14, 14, 14, 14), dtype="float32") = R.reshape(matmul93, R.shape([300, 14, 14, 14, 14]))
            expand_dims111: R.Tensor((300, 14, 14, 1, 14), dtype="float32") = R.expand_dims(einsum30, axis=[3])
            expand_dims112: R.Tensor((300, 14, 1, 14, 14), dtype="float32") = R.expand_dims(einsum31, axis=[2])
            add164: R.Tensor((300, 14, 14, 14, 14), dtype="float32") = R.add(reshape215, expand_dims111)
            add165: R.Tensor((300, 14, 14, 14, 14), dtype="float32") = R.add(add164, expand_dims112)
            reshape216: R.Tensor((300, 196, 196), dtype="float32") = R.reshape(add165, R.shape([300, 196, 196]))
            softmax15: R.Tensor((300, 196, 196), dtype="float32") = R.nn.softmax(reshape216, axis=2)
            matmul94: R.Tensor((300, 196, 64), dtype="float32") = R.matmul(softmax15, unbind47, out_dtype="void")
            reshape217: R.Tensor((25, 12, 14, 14, 64), dtype="float32") = R.reshape(matmul94, R.shape([25, 12, 14, 14, 64]))
            permute_dims197: R.Tensor((25, 14, 14, 12, 64), dtype="float32") = R.permute_dims(reshape217, axes=[0, 2, 3, 1, 4])
            reshape218: R.Tensor((25, 14, 14, 768), dtype="float32") = R.reshape(permute_dims197, R.shape([25, 14, 14, 768]))
            permute_dims198: R.Tensor((768, 768), dtype="float32") = R.permute_dims(vision_encoder_layers_3_attn_proj_weight, axes=[1, 0])
            matmul95: R.Tensor((25, 14, 14, 768), dtype="float32") = R.matmul(reshape218, permute_dims198, out_dtype="void")
            add166: R.Tensor((25, 14, 14, 768), dtype="float32") = R.add(matmul95, vision_encoder_layers_3_attn_proj_bias)
            reshape219: R.Tensor((1, 5, 5, 14, 14, 768), dtype="float32") = R.reshape(add166, R.shape([1, 5, 5, 14, 14, 768]))
            permute_dims199: R.Tensor((1, 5, 14, 5, 14, 768), dtype="float32") = R.permute_dims(reshape219, axes=[0, 1, 3, 2, 4, 5])
            reshape220: R.Tensor((1, 70, 70, 768), dtype="float32") = R.reshape(permute_dims199, R.shape([1, 70, 70, 768]))
            strided_slice12: R.Tensor((1, 64, 64, 768), dtype="float32") = R.strided_slice(reshape220, axes=[1, 2], begin=[0, 0], end=[64, 64], strides=None, assume_inbound=False)
            add167: R.Tensor((1, 64, 64, 768), dtype="float32") = R.add(add160, strided_slice12)
            layer_norm31: R.Tensor((1, 64, 64, 768), dtype="float32") = R.nn.layer_norm(add167, vision_encoder_layers_3_layer_norm2_weight, vision_encoder_layers_3_layer_norm2_bias, axes=[3], epsilon=9.9999999999999995e-07, center=True, scale=True)
            permute_dims200: R.Tensor((768, 3072), dtype="float32") = R.permute_dims(vision_encoder_layers_3_mlp_lin1_weight, axes=[1, 0])
            matmul96: R.Tensor((1, 64, 64, 3072), dtype="float32") = R.matmul(layer_norm31, permute_dims200, out_dtype="void")
            add168: R.Tensor((1, 64, 64, 3072), dtype="float32") = R.add(matmul96, vision_encoder_layers_3_mlp_lin1_bias)
            gelu15: R.Tensor((1, 64, 64, 3072), dtype="float32") = R.nn.gelu(add168)
            permute_dims201: R.Tensor((3072, 768), dtype="float32") = R.permute_dims(vision_encoder_layers_3_mlp_lin2_weight, axes=[1, 0])
            matmul97: R.Tensor((1, 64, 64, 768), dtype="float32") = R.matmul(gelu15, permute_dims201, out_dtype="void")
            add169: R.Tensor((1, 64, 64, 768), dtype="float32") = R.add(matmul97, vision_encoder_layers_3_mlp_lin2_bias)
            add170: R.Tensor((1, 64, 64, 768), dtype="float32") = R.add(add167, add169)
            layer_norm32: R.Tensor((1, 64, 64, 768), dtype="float32") = R.nn.layer_norm(add170, vision_encoder_layers_4_layer_norm1_weight, vision_encoder_layers_4_layer_norm1_bias, axes=[3], epsilon=9.9999999999999995e-07, center=True, scale=True)
            pad11: R.Tensor((1, 70, 70, 768), dtype="float32") = R.nn.pad(layer_norm32, [0, 0, 0, 6, 0, 6, 0, 0], R.const(0, "int32"), pad_mode="constant")
            reshape221: R.Tensor((1, 5, 14, 5, 14, 768), dtype="float32") = R.reshape(pad11, R.shape([1, 5, 14, 5, 14, 768]))
            permute_dims202: R.Tensor((1, 5, 5, 14, 14, 768), dtype="float32") = R.permute_dims(reshape221, axes=[0, 1, 3, 2, 4, 5])
            reshape222: R.Tensor((25, 14, 14, 768), dtype="float32") = R.reshape(permute_dims202, R.shape([25, 14, 14, 768]))
            permute_dims203: R.Tensor((768, 2304), dtype="float32") = R.permute_dims(vision_encoder_layers_4_attn_qkv_weight, axes=[1, 0])
            matmul98: R.Tensor((25, 14, 14, 2304), dtype="float32") = R.matmul(reshape222, permute_dims203, out_dtype="void")
            add171: R.Tensor((25, 14, 14, 2304), dtype="float32") = R.add(matmul98, vision_encoder_layers_4_attn_qkv_bias)
            reshape223: R.Tensor((25, 196, 3, 12, 64), dtype="float32") = R.reshape(add171, R.shape([25, 196, 3, 12, 64]))
            permute_dims204: R.Tensor((3, 25, 12, 196, 64), dtype="float32") = R.permute_dims(reshape223, axes=[2, 0, 3, 1, 4])
            reshape224: R.Tensor((3, 300, 196, 64), dtype="float32") = R.reshape(permute_dims204, R.shape([3, 300, 196, 64]))
            split16: R.Tuple(R.Tensor((1, 300, 196, 64), dtype="float32"), R.Tensor((1, 300, 196, 64), dtype="float32"), R.Tensor((1, 300, 196, 64), dtype="float32")) = R.split(reshape224, indices_or_sections=3, axis=0)
            split_016: R.Tensor((1, 300, 196, 64), dtype="float32") = split16[0]
            split_116: R.Tensor((1, 300, 196, 64), dtype="float32") = split16[1]
            split_216: R.Tensor((1, 300, 196, 64), dtype="float32") = split16[2]
            unbind48: R.Tensor((300, 196, 64), dtype="float32") = R.squeeze(split_016, axis=[0])
            unbind49: R.Tensor((300, 196, 64), dtype="float32") = R.squeeze(split_116, axis=[0])
            unbind50: R.Tensor((300, 196, 64), dtype="float32") = R.squeeze(split_216, axis=[0])
            mul85: R.Tensor((300, 196, 64), dtype="float32") = R.multiply(unbind48, R.const(0.125, "float32"))
            permute_dims205: R.Tensor((300, 64, 196), dtype="float32") = R.permute_dims(unbind49, axes=[0, 2, 1])
            matmul99: R.Tensor((300, 196, 196), dtype="float32") = R.matmul(mul85, permute_dims205, out_dtype="void")
            reshape225: R.Tensor((1, 27, 64), dtype="float32") = R.reshape(vision_encoder_layers_4_attn_rel_pos_h, R.shape([1, 27, 64]))
            permute_dims206: R.Tensor((1, 64, 27), dtype="float32") = R.permute_dims(reshape225, axes=[0, 2, 1])
            reshape226: R.Tensor((64, 27), dtype="float32") = R.reshape(permute_dims206, R.shape([64, 27]))
            permute_dims207: R.Tensor((27, 64), dtype="float32") = R.permute_dims(reshape226, axes=[1, 0])
            take32: R.Tensor((14, 14, 64), dtype="float32") = R.take(permute_dims207, add132, axis=0)
            reshape227: R.Tensor((1, 27, 64), dtype="float32") = R.reshape(vision_encoder_layers_4_attn_rel_pos_w, R.shape([1, 27, 64]))
            permute_dims208: R.Tensor((1, 64, 27), dtype="float32") = R.permute_dims(reshape227, axes=[0, 2, 1])
            reshape228: R.Tensor((64, 27), dtype="float32") = R.reshape(permute_dims208, R.shape([64, 27]))
            permute_dims209: R.Tensor((27, 64), dtype="float32") = R.permute_dims(reshape228, axes=[1, 0])
            take33: R.Tensor((14, 14, 64), dtype="float32") = R.take(permute_dims209, add132, axis=0)
            reshape229: R.Tensor((300, 14, 14, 64), dtype="float32") = R.reshape(unbind48, R.shape([300, 14, 14, 64]))
            einsum32: R.Tensor((300, 14, 14, 14), dtype="float32") = R.einsum((reshape229, take32), subscripts="bhwc,hkc->bhwk")
            einsum33: R.Tensor((300, 14, 14, 14), dtype="float32") = R.einsum((reshape229, take33), subscripts="bhwc,wkc->bhwk")
            reshape230: R.Tensor((300, 14, 14, 14, 14), dtype="float32") = R.reshape(matmul99, R.shape([300, 14, 14, 14, 14]))
            expand_dims117: R.Tensor((300, 14, 14, 1, 14), dtype="float32") = R.expand_dims(einsum32, axis=[3])
            expand_dims118: R.Tensor((300, 14, 1, 14, 14), dtype="float32") = R.expand_dims(einsum33, axis=[2])
            add174: R.Tensor((300, 14, 14, 14, 14), dtype="float32") = R.add(reshape230, expand_dims117)
            add175: R.Tensor((300, 14, 14, 14, 14), dtype="float32") = R.add(add174, expand_dims118)
            reshape231: R.Tensor((300, 196, 196), dtype="float32") = R.reshape(add175, R.shape([300, 196, 196]))
            softmax16: R.Tensor((300, 196, 196), dtype="float32") = R.nn.softmax(reshape231, axis=2)
            matmul100: R.Tensor((300, 196, 64), dtype="float32") = R.matmul(softmax16, unbind50, out_dtype="void")
            reshape232: R.Tensor((25, 12, 14, 14, 64), dtype="float32") = R.reshape(matmul100, R.shape([25, 12, 14, 14, 64]))
            permute_dims210: R.Tensor((25, 14, 14, 12, 64), dtype="float32") = R.permute_dims(reshape232, axes=[0, 2, 3, 1, 4])
            reshape233: R.Tensor((25, 14, 14, 768), dtype="float32") = R.reshape(permute_dims210, R.shape([25, 14, 14, 768]))
            permute_dims211: R.Tensor((768, 768), dtype="float32") = R.permute_dims(vision_encoder_layers_4_attn_proj_weight, axes=[1, 0])
            matmul101: R.Tensor((25, 14, 14, 768), dtype="float32") = R.matmul(reshape233, permute_dims211, out_dtype="void")
            add176: R.Tensor((25, 14, 14, 768), dtype="float32") = R.add(matmul101, vision_encoder_layers_4_attn_proj_bias)
            reshape234: R.Tensor((1, 5, 5, 14, 14, 768), dtype="float32") = R.reshape(add176, R.shape([1, 5, 5, 14, 14, 768]))
            permute_dims212: R.Tensor((1, 5, 14, 5, 14, 768), dtype="float32") = R.permute_dims(reshape234, axes=[0, 1, 3, 2, 4, 5])
            reshape235: R.Tensor((1, 70, 70, 768), dtype="float32") = R.reshape(permute_dims212, R.shape([1, 70, 70, 768]))
            strided_slice13: R.Tensor((1, 64, 64, 768), dtype="float32") = R.strided_slice(reshape235, axes=[1, 2], begin=[0, 0], end=[64, 64], strides=None, assume_inbound=False)
            add177: R.Tensor((1, 64, 64, 768), dtype="float32") = R.add(add170, strided_slice13)
            layer_norm33: R.Tensor((1, 64, 64, 768), dtype="float32") = R.nn.layer_norm(add177, vision_encoder_layers_4_layer_norm2_weight, vision_encoder_layers_4_layer_norm2_bias, axes=[3], epsilon=9.9999999999999995e-07, center=True, scale=True)
            permute_dims213: R.Tensor((768, 3072), dtype="float32") = R.permute_dims(vision_encoder_layers_4_mlp_lin1_weight, axes=[1, 0])
            matmul102: R.Tensor((1, 64, 64, 3072), dtype="float32") = R.matmul(layer_norm33, permute_dims213, out_dtype="void")
            add178: R.Tensor((1, 64, 64, 3072), dtype="float32") = R.add(matmul102, vision_encoder_layers_4_mlp_lin1_bias)
            gelu16: R.Tensor((1, 64, 64, 3072), dtype="float32") = R.nn.gelu(add178)
            permute_dims214: R.Tensor((3072, 768), dtype="float32") = R.permute_dims(vision_encoder_layers_4_mlp_lin2_weight, axes=[1, 0])
            matmul103: R.Tensor((1, 64, 64, 768), dtype="float32") = R.matmul(gelu16, permute_dims214, out_dtype="void")
            add179: R.Tensor((1, 64, 64, 768), dtype="float32") = R.add(matmul103, vision_encoder_layers_4_mlp_lin2_bias)
            add180: R.Tensor((1, 64, 64, 768), dtype="float32") = R.add(add177, add179)
            layer_norm34: R.Tensor((1, 64, 64, 768), dtype="float32") = R.nn.layer_norm(add180, vision_encoder_layers_5_layer_norm1_weight, vision_encoder_layers_5_layer_norm1_bias, axes=[3], epsilon=9.9999999999999995e-07, center=True, scale=True)
            permute_dims215: R.Tensor((768, 2304), dtype="float32") = R.permute_dims(vision_encoder_layers_5_attn_qkv_weight, axes=[1, 0])
            matmul104: R.Tensor((1, 64, 64, 2304), dtype="float32") = R.matmul(layer_norm34, permute_dims215, out_dtype="void")
            add181: R.Tensor((1, 64, 64, 2304), dtype="float32") = R.add(matmul104, vision_encoder_layers_5_attn_qkv_bias)
            reshape236: R.Tensor((1, 4096, 3, 12, 64), dtype="float32") = R.reshape(add181, R.shape([1, 4096, 3, 12, 64]))
            permute_dims216: R.Tensor((3, 1, 12, 4096, 64), dtype="float32") = R.permute_dims(reshape236, axes=[2, 0, 3, 1, 4])
            reshape237: R.Tensor((3, 12, 4096, 64), dtype="float32") = R.reshape(permute_dims216, R.shape([3, 12, 4096, 64]))
            split17: R.Tuple(R.Tensor((1, 12, 4096, 64), dtype="float32"), R.Tensor((1, 12, 4096, 64), dtype="float32"), R.Tensor((1, 12, 4096, 64), dtype="float32")) = R.split(reshape237, indices_or_sections=3, axis=0)
            split_017: R.Tensor((1, 12, 4096, 64), dtype="float32") = split17[0]
            split_117: R.Tensor((1, 12, 4096, 64), dtype="float32") = split17[1]
            split_217: R.Tensor((1, 12, 4096, 64), dtype="float32") = split17[2]
            unbind51: R.Tensor((12, 4096, 64), dtype="float32") = R.squeeze(split_017, axis=[0])
            unbind52: R.Tensor((12, 4096, 64), dtype="float32") = R.squeeze(split_117, axis=[0])
            unbind53: R.Tensor((12, 4096, 64), dtype="float32") = R.squeeze(split_217, axis=[0])
            mul90: R.Tensor((12, 4096, 64), dtype="float32") = R.multiply(unbind51, R.const(0.125, "float32"))
            permute_dims217: R.Tensor((12, 64, 4096), dtype="float32") = R.permute_dims(unbind52, axes=[0, 2, 1])
            matmul105: R.Tensor((12, 4096, 4096), dtype="float32") = R.matmul(mul90, permute_dims217, out_dtype="void")
            reshape238: R.Tensor((1, 127, 64), dtype="float32") = R.reshape(vision_encoder_layers_5_attn_rel_pos_h, R.shape([1, 127, 64]))
            permute_dims218: R.Tensor((1, 64, 127), dtype="float32") = R.permute_dims(reshape238, axes=[0, 2, 1])
            reshape239: R.Tensor((64, 127), dtype="float32") = R.reshape(permute_dims218, R.shape([64, 127]))
            permute_dims219: R.Tensor((127, 64), dtype="float32") = R.permute_dims(reshape239, axes=[1, 0])
            take34: R.Tensor((64, 64, 64), dtype="float32") = R.take(permute_dims219, add152, axis=0)
            reshape240: R.Tensor((1, 127, 64), dtype="float32") = R.reshape(vision_encoder_layers_5_attn_rel_pos_w, R.shape([1, 127, 64]))
            permute_dims220: R.Tensor((1, 64, 127), dtype="float32") = R.permute_dims(reshape240, axes=[0, 2, 1])
            reshape241: R.Tensor((64, 127), dtype="float32") = R.reshape(permute_dims220, R.shape([64, 127]))
            permute_dims221: R.Tensor((127, 64), dtype="float32") = R.permute_dims(reshape241, axes=[1, 0])
            take35: R.Tensor((64, 64, 64), dtype="float32") = R.take(permute_dims221, add152, axis=0)
            reshape242: R.Tensor((12, 64, 64, 64), dtype="float32") = R.reshape(unbind51, R.shape([12, 64, 64, 64]))
            einsum34: R.Tensor((12, 64, 64, 64), dtype="float32") = R.einsum((reshape242, take34), subscripts="bhwc,hkc->bhwk")
            einsum35: R.Tensor((12, 64, 64, 64), dtype="float32") = R.einsum((reshape242, take35), subscripts="bhwc,wkc->bhwk")
            reshape243: R.Tensor((12, 64, 64, 64, 64), dtype="float32") = R.reshape(matmul105, R.shape([12, 64, 64, 64, 64]))
            expand_dims123: R.Tensor((12, 64, 64, 1, 64), dtype="float32") = R.expand_dims(einsum34, axis=[3])
            expand_dims124: R.Tensor((12, 64, 1, 64, 64), dtype="float32") = R.expand_dims(einsum35, axis=[2])
            add184: R.Tensor((12, 64, 64, 64, 64), dtype="float32") = R.add(reshape243, expand_dims123)
            add185: R.Tensor((12, 64, 64, 64, 64), dtype="float32") = R.add(add184, expand_dims124)
            reshape244: R.Tensor((12, 4096, 4096), dtype="float32") = R.reshape(add185, R.shape([12, 4096, 4096]))
            softmax17: R.Tensor((12, 4096, 4096), dtype="float32") = R.nn.softmax(reshape244, axis=2)
            matmul106: R.Tensor((12, 4096, 64), dtype="float32") = R.matmul(softmax17, unbind53, out_dtype="void")
            reshape245: R.Tensor((1, 12, 64, 64, 64), dtype="float32") = R.reshape(matmul106, R.shape([1, 12, 64, 64, 64]))
            permute_dims222: R.Tensor((1, 64, 64, 12, 64), dtype="float32") = R.permute_dims(reshape245, axes=[0, 2, 3, 1, 4])
            reshape246: R.Tensor((1, 64, 64, 768), dtype="float32") = R.reshape(permute_dims222, R.shape([1, 64, 64, 768]))
            permute_dims223: R.Tensor((768, 768), dtype="float32") = R.permute_dims(vision_encoder_layers_5_attn_proj_weight, axes=[1, 0])
            matmul107: R.Tensor((1, 64, 64, 768), dtype="float32") = R.matmul(reshape246, permute_dims223, out_dtype="void")
            add186: R.Tensor((1, 64, 64, 768), dtype="float32") = R.add(matmul107, vision_encoder_layers_5_attn_proj_bias)
            add187: R.Tensor((1, 64, 64, 768), dtype="float32") = R.add(add180, add186)
            layer_norm35: R.Tensor((1, 64, 64, 768), dtype="float32") = R.nn.layer_norm(add187, vision_encoder_layers_5_layer_norm2_weight, vision_encoder_layers_5_layer_norm2_bias, axes=[3], epsilon=9.9999999999999995e-07, center=True, scale=True)
            permute_dims224: R.Tensor((768, 3072), dtype="float32") = R.permute_dims(vision_encoder_layers_5_mlp_lin1_weight, axes=[1, 0])
            matmul108: R.Tensor((1, 64, 64, 3072), dtype="float32") = R.matmul(layer_norm35, permute_dims224, out_dtype="void")
            add188: R.Tensor((1, 64, 64, 3072), dtype="float32") = R.add(matmul108, vision_encoder_layers_5_mlp_lin1_bias)
            gelu17: R.Tensor((1, 64, 64, 3072), dtype="float32") = R.nn.gelu(add188)
            permute_dims225: R.Tensor((3072, 768), dtype="float32") = R.permute_dims(vision_encoder_layers_5_mlp_lin2_weight, axes=[1, 0])
            matmul109: R.Tensor((1, 64, 64, 768), dtype="float32") = R.matmul(gelu17, permute_dims225, out_dtype="void")
            add189: R.Tensor((1, 64, 64, 768), dtype="float32") = R.add(matmul109, vision_encoder_layers_5_mlp_lin2_bias)
            add190: R.Tensor((1, 64, 64, 768), dtype="float32") = R.add(add187, add189)
            layer_norm36: R.Tensor((1, 64, 64, 768), dtype="float32") = R.nn.layer_norm(add190, vision_encoder_layers_6_layer_norm1_weight, vision_encoder_layers_6_layer_norm1_bias, axes=[3], epsilon=9.9999999999999995e-07, center=True, scale=True)
            pad12: R.Tensor((1, 70, 70, 768), dtype="float32") = R.nn.pad(layer_norm36, [0, 0, 0, 6, 0, 6, 0, 0], R.const(0, "int32"), pad_mode="constant")
            reshape247: R.Tensor((1, 5, 14, 5, 14, 768), dtype="float32") = R.reshape(pad12, R.shape([1, 5, 14, 5, 14, 768]))
            permute_dims226: R.Tensor((1, 5, 5, 14, 14, 768), dtype="float32") = R.permute_dims(reshape247, axes=[0, 1, 3, 2, 4, 5])
            reshape248: R.Tensor((25, 14, 14, 768), dtype="float32") = R.reshape(permute_dims226, R.shape([25, 14, 14, 768]))
            permute_dims227: R.Tensor((768, 2304), dtype="float32") = R.permute_dims(vision_encoder_layers_6_attn_qkv_weight, axes=[1, 0])
            matmul110: R.Tensor((25, 14, 14, 2304), dtype="float32") = R.matmul(reshape248, permute_dims227, out_dtype="void")
            add191: R.Tensor((25, 14, 14, 2304), dtype="float32") = R.add(matmul110, vision_encoder_layers_6_attn_qkv_bias)
            reshape249: R.Tensor((25, 196, 3, 12, 64), dtype="float32") = R.reshape(add191, R.shape([25, 196, 3, 12, 64]))
            permute_dims228: R.Tensor((3, 25, 12, 196, 64), dtype="float32") = R.permute_dims(reshape249, axes=[2, 0, 3, 1, 4])
            reshape250: R.Tensor((3, 300, 196, 64), dtype="float32") = R.reshape(permute_dims228, R.shape([3, 300, 196, 64]))
            split18: R.Tuple(R.Tensor((1, 300, 196, 64), dtype="float32"), R.Tensor((1, 300, 196, 64), dtype="float32"), R.Tensor((1, 300, 196, 64), dtype="float32")) = R.split(reshape250, indices_or_sections=3, axis=0)
            split_018: R.Tensor((1, 300, 196, 64), dtype="float32") = split18[0]
            split_118: R.Tensor((1, 300, 196, 64), dtype="float32") = split18[1]
            split_218: R.Tensor((1, 300, 196, 64), dtype="float32") = split18[2]
            unbind54: R.Tensor((300, 196, 64), dtype="float32") = R.squeeze(split_018, axis=[0])
            unbind55: R.Tensor((300, 196, 64), dtype="float32") = R.squeeze(split_118, axis=[0])
            unbind56: R.Tensor((300, 196, 64), dtype="float32") = R.squeeze(split_218, axis=[0])
            mul95: R.Tensor((300, 196, 64), dtype="float32") = R.multiply(unbind54, R.const(0.125, "float32"))
            permute_dims229: R.Tensor((300, 64, 196), dtype="float32") = R.permute_dims(unbind55, axes=[0, 2, 1])
            matmul111: R.Tensor((300, 196, 196), dtype="float32") = R.matmul(mul95, permute_dims229, out_dtype="void")
            reshape251: R.Tensor((1, 27, 64), dtype="float32") = R.reshape(vision_encoder_layers_6_attn_rel_pos_h, R.shape([1, 27, 64]))
            permute_dims230: R.Tensor((1, 64, 27), dtype="float32") = R.permute_dims(reshape251, axes=[0, 2, 1])
            reshape252: R.Tensor((64, 27), dtype="float32") = R.reshape(permute_dims230, R.shape([64, 27]))
            permute_dims231: R.Tensor((27, 64), dtype="float32") = R.permute_dims(reshape252, axes=[1, 0])
            take36: R.Tensor((14, 14, 64), dtype="float32") = R.take(permute_dims231, add132, axis=0)
            reshape253: R.Tensor((1, 27, 64), dtype="float32") = R.reshape(vision_encoder_layers_6_attn_rel_pos_w, R.shape([1, 27, 64]))
            permute_dims232: R.Tensor((1, 64, 27), dtype="float32") = R.permute_dims(reshape253, axes=[0, 2, 1])
            reshape254: R.Tensor((64, 27), dtype="float32") = R.reshape(permute_dims232, R.shape([64, 27]))
            permute_dims233: R.Tensor((27, 64), dtype="float32") = R.permute_dims(reshape254, axes=[1, 0])
            take37: R.Tensor((14, 14, 64), dtype="float32") = R.take(permute_dims233, add132, axis=0)
            reshape255: R.Tensor((300, 14, 14, 64), dtype="float32") = R.reshape(unbind54, R.shape([300, 14, 14, 64]))
            einsum36: R.Tensor((300, 14, 14, 14), dtype="float32") = R.einsum((reshape255, take36), subscripts="bhwc,hkc->bhwk")
            einsum37: R.Tensor((300, 14, 14, 14), dtype="float32") = R.einsum((reshape255, take37), subscripts="bhwc,wkc->bhwk")
            reshape256: R.Tensor((300, 14, 14, 14, 14), dtype="float32") = R.reshape(matmul111, R.shape([300, 14, 14, 14, 14]))
            expand_dims129: R.Tensor((300, 14, 14, 1, 14), dtype="float32") = R.expand_dims(einsum36, axis=[3])
            expand_dims130: R.Tensor((300, 14, 1, 14, 14), dtype="float32") = R.expand_dims(einsum37, axis=[2])
            add194: R.Tensor((300, 14, 14, 14, 14), dtype="float32") = R.add(reshape256, expand_dims129)
            add195: R.Tensor((300, 14, 14, 14, 14), dtype="float32") = R.add(add194, expand_dims130)
            reshape257: R.Tensor((300, 196, 196), dtype="float32") = R.reshape(add195, R.shape([300, 196, 196]))
            softmax18: R.Tensor((300, 196, 196), dtype="float32") = R.nn.softmax(reshape257, axis=2)
            matmul112: R.Tensor((300, 196, 64), dtype="float32") = R.matmul(softmax18, unbind56, out_dtype="void")
            reshape258: R.Tensor((25, 12, 14, 14, 64), dtype="float32") = R.reshape(matmul112, R.shape([25, 12, 14, 14, 64]))
            permute_dims234: R.Tensor((25, 14, 14, 12, 64), dtype="float32") = R.permute_dims(reshape258, axes=[0, 2, 3, 1, 4])
            reshape259: R.Tensor((25, 14, 14, 768), dtype="float32") = R.reshape(permute_dims234, R.shape([25, 14, 14, 768]))
            permute_dims235: R.Tensor((768, 768), dtype="float32") = R.permute_dims(vision_encoder_layers_6_attn_proj_weight, axes=[1, 0])
            matmul113: R.Tensor((25, 14, 14, 768), dtype="float32") = R.matmul(reshape259, permute_dims235, out_dtype="void")
            add196: R.Tensor((25, 14, 14, 768), dtype="float32") = R.add(matmul113, vision_encoder_layers_6_attn_proj_bias)
            reshape260: R.Tensor((1, 5, 5, 14, 14, 768), dtype="float32") = R.reshape(add196, R.shape([1, 5, 5, 14, 14, 768]))
            permute_dims236: R.Tensor((1, 5, 14, 5, 14, 768), dtype="float32") = R.permute_dims(reshape260, axes=[0, 1, 3, 2, 4, 5])
            reshape261: R.Tensor((1, 70, 70, 768), dtype="float32") = R.reshape(permute_dims236, R.shape([1, 70, 70, 768]))
            strided_slice14: R.Tensor((1, 64, 64, 768), dtype="float32") = R.strided_slice(reshape261, axes=[1, 2], begin=[0, 0], end=[64, 64], strides=None, assume_inbound=False)
            add197: R.Tensor((1, 64, 64, 768), dtype="float32") = R.add(add190, strided_slice14)
            layer_norm37: R.Tensor((1, 64, 64, 768), dtype="float32") = R.nn.layer_norm(add197, vision_encoder_layers_6_layer_norm2_weight, vision_encoder_layers_6_layer_norm2_bias, axes=[3], epsilon=9.9999999999999995e-07, center=True, scale=True)
            permute_dims237: R.Tensor((768, 3072), dtype="float32") = R.permute_dims(vision_encoder_layers_6_mlp_lin1_weight, axes=[1, 0])
            matmul114: R.Tensor((1, 64, 64, 3072), dtype="float32") = R.matmul(layer_norm37, permute_dims237, out_dtype="void")
            add198: R.Tensor((1, 64, 64, 3072), dtype="float32") = R.add(matmul114, vision_encoder_layers_6_mlp_lin1_bias)
            gelu18: R.Tensor((1, 64, 64, 3072), dtype="float32") = R.nn.gelu(add198)
            permute_dims238: R.Tensor((3072, 768), dtype="float32") = R.permute_dims(vision_encoder_layers_6_mlp_lin2_weight, axes=[1, 0])
            matmul115: R.Tensor((1, 64, 64, 768), dtype="float32") = R.matmul(gelu18, permute_dims238, out_dtype="void")
            add199: R.Tensor((1, 64, 64, 768), dtype="float32") = R.add(matmul115, vision_encoder_layers_6_mlp_lin2_bias)
            add200: R.Tensor((1, 64, 64, 768), dtype="float32") = R.add(add197, add199)
            layer_norm38: R.Tensor((1, 64, 64, 768), dtype="float32") = R.nn.layer_norm(add200, vision_encoder_layers_7_layer_norm1_weight, vision_encoder_layers_7_layer_norm1_bias, axes=[3], epsilon=9.9999999999999995e-07, center=True, scale=True)
            pad13: R.Tensor((1, 70, 70, 768), dtype="float32") = R.nn.pad(layer_norm38, [0, 0, 0, 6, 0, 6, 0, 0], R.const(0, "int32"), pad_mode="constant")
            reshape262: R.Tensor((1, 5, 14, 5, 14, 768), dtype="float32") = R.reshape(pad13, R.shape([1, 5, 14, 5, 14, 768]))
            permute_dims239: R.Tensor((1, 5, 5, 14, 14, 768), dtype="float32") = R.permute_dims(reshape262, axes=[0, 1, 3, 2, 4, 5])
            reshape263: R.Tensor((25, 14, 14, 768), dtype="float32") = R.reshape(permute_dims239, R.shape([25, 14, 14, 768]))
            permute_dims240: R.Tensor((768, 2304), dtype="float32") = R.permute_dims(vision_encoder_layers_7_attn_qkv_weight, axes=[1, 0])
            matmul116: R.Tensor((25, 14, 14, 2304), dtype="float32") = R.matmul(reshape263, permute_dims240, out_dtype="void")
            add201: R.Tensor((25, 14, 14, 2304), dtype="float32") = R.add(matmul116, vision_encoder_layers_7_attn_qkv_bias)
            reshape264: R.Tensor((25, 196, 3, 12, 64), dtype="float32") = R.reshape(add201, R.shape([25, 196, 3, 12, 64]))
            permute_dims241: R.Tensor((3, 25, 12, 196, 64), dtype="float32") = R.permute_dims(reshape264, axes=[2, 0, 3, 1, 4])
            reshape265: R.Tensor((3, 300, 196, 64), dtype="float32") = R.reshape(permute_dims241, R.shape([3, 300, 196, 64]))
            split19: R.Tuple(R.Tensor((1, 300, 196, 64), dtype="float32"), R.Tensor((1, 300, 196, 64), dtype="float32"), R.Tensor((1, 300, 196, 64), dtype="float32")) = R.split(reshape265, indices_or_sections=3, axis=0)
            split_019: R.Tensor((1, 300, 196, 64), dtype="float32") = split19[0]
            split_119: R.Tensor((1, 300, 196, 64), dtype="float32") = split19[1]
            split_219: R.Tensor((1, 300, 196, 64), dtype="float32") = split19[2]
            unbind57: R.Tensor((300, 196, 64), dtype="float32") = R.squeeze(split_019, axis=[0])
            unbind58: R.Tensor((300, 196, 64), dtype="float32") = R.squeeze(split_119, axis=[0])
            unbind59: R.Tensor((300, 196, 64), dtype="float32") = R.squeeze(split_219, axis=[0])
            mul100: R.Tensor((300, 196, 64), dtype="float32") = R.multiply(unbind57, R.const(0.125, "float32"))
            permute_dims242: R.Tensor((300, 64, 196), dtype="float32") = R.permute_dims(unbind58, axes=[0, 2, 1])
            matmul117: R.Tensor((300, 196, 196), dtype="float32") = R.matmul(mul100, permute_dims242, out_dtype="void")
            reshape266: R.Tensor((1, 27, 64), dtype="float32") = R.reshape(vision_encoder_layers_7_attn_rel_pos_h, R.shape([1, 27, 64]))
            permute_dims243: R.Tensor((1, 64, 27), dtype="float32") = R.permute_dims(reshape266, axes=[0, 2, 1])
            reshape267: R.Tensor((64, 27), dtype="float32") = R.reshape(permute_dims243, R.shape([64, 27]))
            permute_dims244: R.Tensor((27, 64), dtype="float32") = R.permute_dims(reshape267, axes=[1, 0])
            take38: R.Tensor((14, 14, 64), dtype="float32") = R.take(permute_dims244, add132, axis=0)
            reshape268: R.Tensor((1, 27, 64), dtype="float32") = R.reshape(vision_encoder_layers_7_attn_rel_pos_w, R.shape([1, 27, 64]))
            permute_dims245: R.Tensor((1, 64, 27), dtype="float32") = R.permute_dims(reshape268, axes=[0, 2, 1])
            reshape269: R.Tensor((64, 27), dtype="float32") = R.reshape(permute_dims245, R.shape([64, 27]))
            permute_dims246: R.Tensor((27, 64), dtype="float32") = R.permute_dims(reshape269, axes=[1, 0])
            take39: R.Tensor((14, 14, 64), dtype="float32") = R.take(permute_dims246, add132, axis=0)
            reshape270: R.Tensor((300, 14, 14, 64), dtype="float32") = R.reshape(unbind57, R.shape([300, 14, 14, 64]))
            einsum38: R.Tensor((300, 14, 14, 14), dtype="float32") = R.einsum((reshape270, take38), subscripts="bhwc,hkc->bhwk")
            einsum39: R.Tensor((300, 14, 14, 14), dtype="float32") = R.einsum((reshape270, take39), subscripts="bhwc,wkc->bhwk")
            reshape271: R.Tensor((300, 14, 14, 14, 14), dtype="float32") = R.reshape(matmul117, R.shape([300, 14, 14, 14, 14]))
            expand_dims135: R.Tensor((300, 14, 14, 1, 14), dtype="float32") = R.expand_dims(einsum38, axis=[3])
            expand_dims136: R.Tensor((300, 14, 1, 14, 14), dtype="float32") = R.expand_dims(einsum39, axis=[2])
            add204: R.Tensor((300, 14, 14, 14, 14), dtype="float32") = R.add(reshape271, expand_dims135)
            add205: R.Tensor((300, 14, 14, 14, 14), dtype="float32") = R.add(add204, expand_dims136)
            reshape272: R.Tensor((300, 196, 196), dtype="float32") = R.reshape(add205, R.shape([300, 196, 196]))
            softmax19: R.Tensor((300, 196, 196), dtype="float32") = R.nn.softmax(reshape272, axis=2)
            matmul118: R.Tensor((300, 196, 64), dtype="float32") = R.matmul(softmax19, unbind59, out_dtype="void")
            reshape273: R.Tensor((25, 12, 14, 14, 64), dtype="float32") = R.reshape(matmul118, R.shape([25, 12, 14, 14, 64]))
            permute_dims247: R.Tensor((25, 14, 14, 12, 64), dtype="float32") = R.permute_dims(reshape273, axes=[0, 2, 3, 1, 4])
            reshape274: R.Tensor((25, 14, 14, 768), dtype="float32") = R.reshape(permute_dims247, R.shape([25, 14, 14, 768]))
            permute_dims248: R.Tensor((768, 768), dtype="float32") = R.permute_dims(vision_encoder_layers_7_attn_proj_weight, axes=[1, 0])
            matmul119: R.Tensor((25, 14, 14, 768), dtype="float32") = R.matmul(reshape274, permute_dims248, out_dtype="void")
            add206: R.Tensor((25, 14, 14, 768), dtype="float32") = R.add(matmul119, vision_encoder_layers_7_attn_proj_bias)
            reshape275: R.Tensor((1, 5, 5, 14, 14, 768), dtype="float32") = R.reshape(add206, R.shape([1, 5, 5, 14, 14, 768]))
            permute_dims249: R.Tensor((1, 5, 14, 5, 14, 768), dtype="float32") = R.permute_dims(reshape275, axes=[0, 1, 3, 2, 4, 5])
            reshape276: R.Tensor((1, 70, 70, 768), dtype="float32") = R.reshape(permute_dims249, R.shape([1, 70, 70, 768]))
            strided_slice15: R.Tensor((1, 64, 64, 768), dtype="float32") = R.strided_slice(reshape276, axes=[1, 2], begin=[0, 0], end=[64, 64], strides=None, assume_inbound=False)
            add207: R.Tensor((1, 64, 64, 768), dtype="float32") = R.add(add200, strided_slice15)
            layer_norm39: R.Tensor((1, 64, 64, 768), dtype="float32") = R.nn.layer_norm(add207, vision_encoder_layers_7_layer_norm2_weight, vision_encoder_layers_7_layer_norm2_bias, axes=[3], epsilon=9.9999999999999995e-07, center=True, scale=True)
            permute_dims250: R.Tensor((768, 3072), dtype="float32") = R.permute_dims(vision_encoder_layers_7_mlp_lin1_weight, axes=[1, 0])
            matmul120: R.Tensor((1, 64, 64, 3072), dtype="float32") = R.matmul(layer_norm39, permute_dims250, out_dtype="void")
            add208: R.Tensor((1, 64, 64, 3072), dtype="float32") = R.add(matmul120, vision_encoder_layers_7_mlp_lin1_bias)
            gelu19: R.Tensor((1, 64, 64, 3072), dtype="float32") = R.nn.gelu(add208)
            permute_dims251: R.Tensor((3072, 768), dtype="float32") = R.permute_dims(vision_encoder_layers_7_mlp_lin2_weight, axes=[1, 0])
            matmul121: R.Tensor((1, 64, 64, 768), dtype="float32") = R.matmul(gelu19, permute_dims251, out_dtype="void")
            add209: R.Tensor((1, 64, 64, 768), dtype="float32") = R.add(matmul121, vision_encoder_layers_7_mlp_lin2_bias)
            add210: R.Tensor((1, 64, 64, 768), dtype="float32") = R.add(add207, add209)
            layer_norm40: R.Tensor((1, 64, 64, 768), dtype="float32") = R.nn.layer_norm(add210, vision_encoder_layers_8_layer_norm1_weight, vision_encoder_layers_8_layer_norm1_bias, axes=[3], epsilon=9.9999999999999995e-07, center=True, scale=True)
            permute_dims252: R.Tensor((768, 2304), dtype="float32") = R.permute_dims(vision_encoder_layers_8_attn_qkv_weight, axes=[1, 0])
            matmul122: R.Tensor((1, 64, 64, 2304), dtype="float32") = R.matmul(layer_norm40, permute_dims252, out_dtype="void")
            add211: R.Tensor((1, 64, 64, 2304), dtype="float32") = R.add(matmul122, vision_encoder_layers_8_attn_qkv_bias)
            reshape277: R.Tensor((1, 4096, 3, 12, 64), dtype="float32") = R.reshape(add211, R.shape([1, 4096, 3, 12, 64]))
            permute_dims253: R.Tensor((3, 1, 12, 4096, 64), dtype="float32") = R.permute_dims(reshape277, axes=[2, 0, 3, 1, 4])
            reshape278: R.Tensor((3, 12, 4096, 64), dtype="float32") = R.reshape(permute_dims253, R.shape([3, 12, 4096, 64]))
            split20: R.Tuple(R.Tensor((1, 12, 4096, 64), dtype="float32"), R.Tensor((1, 12, 4096, 64), dtype="float32"), R.Tensor((1, 12, 4096, 64), dtype="float32")) = R.split(reshape278, indices_or_sections=3, axis=0)
            split_020: R.Tensor((1, 12, 4096, 64), dtype="float32") = split20[0]
            split_120: R.Tensor((1, 12, 4096, 64), dtype="float32") = split20[1]
            split_220: R.Tensor((1, 12, 4096, 64), dtype="float32") = split20[2]
            unbind60: R.Tensor((12, 4096, 64), dtype="float32") = R.squeeze(split_020, axis=[0])
            unbind61: R.Tensor((12, 4096, 64), dtype="float32") = R.squeeze(split_120, axis=[0])
            unbind62: R.Tensor((12, 4096, 64), dtype="float32") = R.squeeze(split_220, axis=[0])
            mul105: R.Tensor((12, 4096, 64), dtype="float32") = R.multiply(unbind60, R.const(0.125, "float32"))
            permute_dims254: R.Tensor((12, 64, 4096), dtype="float32") = R.permute_dims(unbind61, axes=[0, 2, 1])
            matmul123: R.Tensor((12, 4096, 4096), dtype="float32") = R.matmul(mul105, permute_dims254, out_dtype="void")
            reshape279: R.Tensor((1, 127, 64), dtype="float32") = R.reshape(vision_encoder_layers_8_attn_rel_pos_h, R.shape([1, 127, 64]))
            permute_dims255: R.Tensor((1, 64, 127), dtype="float32") = R.permute_dims(reshape279, axes=[0, 2, 1])
            reshape280: R.Tensor((64, 127), dtype="float32") = R.reshape(permute_dims255, R.shape([64, 127]))
            permute_dims256: R.Tensor((127, 64), dtype="float32") = R.permute_dims(reshape280, axes=[1, 0])
            take40: R.Tensor((64, 64, 64), dtype="float32") = R.take(permute_dims256, add152, axis=0)
            reshape281: R.Tensor((1, 127, 64), dtype="float32") = R.reshape(vision_encoder_layers_8_attn_rel_pos_w, R.shape([1, 127, 64]))
            permute_dims257: R.Tensor((1, 64, 127), dtype="float32") = R.permute_dims(reshape281, axes=[0, 2, 1])
            reshape282: R.Tensor((64, 127), dtype="float32") = R.reshape(permute_dims257, R.shape([64, 127]))
            permute_dims258: R.Tensor((127, 64), dtype="float32") = R.permute_dims(reshape282, axes=[1, 0])
            take41: R.Tensor((64, 64, 64), dtype="float32") = R.take(permute_dims258, add152, axis=0)
            reshape283: R.Tensor((12, 64, 64, 64), dtype="float32") = R.reshape(unbind60, R.shape([12, 64, 64, 64]))
            einsum40: R.Tensor((12, 64, 64, 64), dtype="float32") = R.einsum((reshape283, take40), subscripts="bhwc,hkc->bhwk")
            einsum41: R.Tensor((12, 64, 64, 64), dtype="float32") = R.einsum((reshape283, take41), subscripts="bhwc,wkc->bhwk")
            reshape284: R.Tensor((12, 64, 64, 64, 64), dtype="float32") = R.reshape(matmul123, R.shape([12, 64, 64, 64, 64]))
            expand_dims141: R.Tensor((12, 64, 64, 1, 64), dtype="float32") = R.expand_dims(einsum40, axis=[3])
            expand_dims142: R.Tensor((12, 64, 1, 64, 64), dtype="float32") = R.expand_dims(einsum41, axis=[2])
            add214: R.Tensor((12, 64, 64, 64, 64), dtype="float32") = R.add(reshape284, expand_dims141)
            add215: R.Tensor((12, 64, 64, 64, 64), dtype="float32") = R.add(add214, expand_dims142)
            reshape285: R.Tensor((12, 4096, 4096), dtype="float32") = R.reshape(add215, R.shape([12, 4096, 4096]))
            softmax20: R.Tensor((12, 4096, 4096), dtype="float32") = R.nn.softmax(reshape285, axis=2)
            matmul124: R.Tensor((12, 4096, 64), dtype="float32") = R.matmul(softmax20, unbind62, out_dtype="void")
            reshape286: R.Tensor((1, 12, 64, 64, 64), dtype="float32") = R.reshape(matmul124, R.shape([1, 12, 64, 64, 64]))
            permute_dims259: R.Tensor((1, 64, 64, 12, 64), dtype="float32") = R.permute_dims(reshape286, axes=[0, 2, 3, 1, 4])
            reshape287: R.Tensor((1, 64, 64, 768), dtype="float32") = R.reshape(permute_dims259, R.shape([1, 64, 64, 768]))
            permute_dims260: R.Tensor((768, 768), dtype="float32") = R.permute_dims(vision_encoder_layers_8_attn_proj_weight, axes=[1, 0])
            matmul125: R.Tensor((1, 64, 64, 768), dtype="float32") = R.matmul(reshape287, permute_dims260, out_dtype="void")
            add216: R.Tensor((1, 64, 64, 768), dtype="float32") = R.add(matmul125, vision_encoder_layers_8_attn_proj_bias)
            add217: R.Tensor((1, 64, 64, 768), dtype="float32") = R.add(add210, add216)
            layer_norm41: R.Tensor((1, 64, 64, 768), dtype="float32") = R.nn.layer_norm(add217, vision_encoder_layers_8_layer_norm2_weight, vision_encoder_layers_8_layer_norm2_bias, axes=[3], epsilon=9.9999999999999995e-07, center=True, scale=True)
            permute_dims261: R.Tensor((768, 3072), dtype="float32") = R.permute_dims(vision_encoder_layers_8_mlp_lin1_weight, axes=[1, 0])
            matmul126: R.Tensor((1, 64, 64, 3072), dtype="float32") = R.matmul(layer_norm41, permute_dims261, out_dtype="void")
            add218: R.Tensor((1, 64, 64, 3072), dtype="float32") = R.add(matmul126, vision_encoder_layers_8_mlp_lin1_bias)
            gelu20: R.Tensor((1, 64, 64, 3072), dtype="float32") = R.nn.gelu(add218)
            permute_dims262: R.Tensor((3072, 768), dtype="float32") = R.permute_dims(vision_encoder_layers_8_mlp_lin2_weight, axes=[1, 0])
            matmul127: R.Tensor((1, 64, 64, 768), dtype="float32") = R.matmul(gelu20, permute_dims262, out_dtype="void")
            add219: R.Tensor((1, 64, 64, 768), dtype="float32") = R.add(matmul127, vision_encoder_layers_8_mlp_lin2_bias)
            add220: R.Tensor((1, 64, 64, 768), dtype="float32") = R.add(add217, add219)
            layer_norm42: R.Tensor((1, 64, 64, 768), dtype="float32") = R.nn.layer_norm(add220, vision_encoder_layers_9_layer_norm1_weight, vision_encoder_layers_9_layer_norm1_bias, axes=[3], epsilon=9.9999999999999995e-07, center=True, scale=True)
            pad14: R.Tensor((1, 70, 70, 768), dtype="float32") = R.nn.pad(layer_norm42, [0, 0, 0, 6, 0, 6, 0, 0], R.const(0, "int32"), pad_mode="constant")
            reshape288: R.Tensor((1, 5, 14, 5, 14, 768), dtype="float32") = R.reshape(pad14, R.shape([1, 5, 14, 5, 14, 768]))
            permute_dims263: R.Tensor((1, 5, 5, 14, 14, 768), dtype="float32") = R.permute_dims(reshape288, axes=[0, 1, 3, 2, 4, 5])
            reshape289: R.Tensor((25, 14, 14, 768), dtype="float32") = R.reshape(permute_dims263, R.shape([25, 14, 14, 768]))
            permute_dims264: R.Tensor((768, 2304), dtype="float32") = R.permute_dims(vision_encoder_layers_9_attn_qkv_weight, axes=[1, 0])
            matmul128: R.Tensor((25, 14, 14, 2304), dtype="float32") = R.matmul(reshape289, permute_dims264, out_dtype="void")
            add221: R.Tensor((25, 14, 14, 2304), dtype="float32") = R.add(matmul128, vision_encoder_layers_9_attn_qkv_bias)
            reshape290: R.Tensor((25, 196, 3, 12, 64), dtype="float32") = R.reshape(add221, R.shape([25, 196, 3, 12, 64]))
            permute_dims265: R.Tensor((3, 25, 12, 196, 64), dtype="float32") = R.permute_dims(reshape290, axes=[2, 0, 3, 1, 4])
            reshape291: R.Tensor((3, 300, 196, 64), dtype="float32") = R.reshape(permute_dims265, R.shape([3, 300, 196, 64]))
            split21: R.Tuple(R.Tensor((1, 300, 196, 64), dtype="float32"), R.Tensor((1, 300, 196, 64), dtype="float32"), R.Tensor((1, 300, 196, 64), dtype="float32")) = R.split(reshape291, indices_or_sections=3, axis=0)
            split_021: R.Tensor((1, 300, 196, 64), dtype="float32") = split21[0]
            split_121: R.Tensor((1, 300, 196, 64), dtype="float32") = split21[1]
            split_221: R.Tensor((1, 300, 196, 64), dtype="float32") = split21[2]
            unbind63: R.Tensor((300, 196, 64), dtype="float32") = R.squeeze(split_021, axis=[0])
            unbind64: R.Tensor((300, 196, 64), dtype="float32") = R.squeeze(split_121, axis=[0])
            unbind65: R.Tensor((300, 196, 64), dtype="float32") = R.squeeze(split_221, axis=[0])
            mul110: R.Tensor((300, 196, 64), dtype="float32") = R.multiply(unbind63, R.const(0.125, "float32"))
            permute_dims266: R.Tensor((300, 64, 196), dtype="float32") = R.permute_dims(unbind64, axes=[0, 2, 1])
            matmul129: R.Tensor((300, 196, 196), dtype="float32") = R.matmul(mul110, permute_dims266, out_dtype="void")
            reshape292: R.Tensor((1, 27, 64), dtype="float32") = R.reshape(vision_encoder_layers_9_attn_rel_pos_h, R.shape([1, 27, 64]))
            permute_dims267: R.Tensor((1, 64, 27), dtype="float32") = R.permute_dims(reshape292, axes=[0, 2, 1])
            reshape293: R.Tensor((64, 27), dtype="float32") = R.reshape(permute_dims267, R.shape([64, 27]))
            permute_dims268: R.Tensor((27, 64), dtype="float32") = R.permute_dims(reshape293, axes=[1, 0])
            take42: R.Tensor((14, 14, 64), dtype="float32") = R.take(permute_dims268, add132, axis=0)
            reshape294: R.Tensor((1, 27, 64), dtype="float32") = R.reshape(vision_encoder_layers_9_attn_rel_pos_w, R.shape([1, 27, 64]))
            permute_dims269: R.Tensor((1, 64, 27), dtype="float32") = R.permute_dims(reshape294, axes=[0, 2, 1])
            reshape295: R.Tensor((64, 27), dtype="float32") = R.reshape(permute_dims269, R.shape([64, 27]))
            permute_dims270: R.Tensor((27, 64), dtype="float32") = R.permute_dims(reshape295, axes=[1, 0])
            take43: R.Tensor((14, 14, 64), dtype="float32") = R.take(permute_dims270, add132, axis=0)
            reshape296: R.Tensor((300, 14, 14, 64), dtype="float32") = R.reshape(unbind63, R.shape([300, 14, 14, 64]))
            einsum42: R.Tensor((300, 14, 14, 14), dtype="float32") = R.einsum((reshape296, take42), subscripts="bhwc,hkc->bhwk")
            einsum43: R.Tensor((300, 14, 14, 14), dtype="float32") = R.einsum((reshape296, take43), subscripts="bhwc,wkc->bhwk")
            reshape297: R.Tensor((300, 14, 14, 14, 14), dtype="float32") = R.reshape(matmul129, R.shape([300, 14, 14, 14, 14]))
            expand_dims147: R.Tensor((300, 14, 14, 1, 14), dtype="float32") = R.expand_dims(einsum42, axis=[3])
            expand_dims148: R.Tensor((300, 14, 1, 14, 14), dtype="float32") = R.expand_dims(einsum43, axis=[2])
            add224: R.Tensor((300, 14, 14, 14, 14), dtype="float32") = R.add(reshape297, expand_dims147)
            add225: R.Tensor((300, 14, 14, 14, 14), dtype="float32") = R.add(add224, expand_dims148)
            reshape298: R.Tensor((300, 196, 196), dtype="float32") = R.reshape(add225, R.shape([300, 196, 196]))
            softmax21: R.Tensor((300, 196, 196), dtype="float32") = R.nn.softmax(reshape298, axis=2)
            matmul130: R.Tensor((300, 196, 64), dtype="float32") = R.matmul(softmax21, unbind65, out_dtype="void")
            reshape299: R.Tensor((25, 12, 14, 14, 64), dtype="float32") = R.reshape(matmul130, R.shape([25, 12, 14, 14, 64]))
            permute_dims271: R.Tensor((25, 14, 14, 12, 64), dtype="float32") = R.permute_dims(reshape299, axes=[0, 2, 3, 1, 4])
            reshape300: R.Tensor((25, 14, 14, 768), dtype="float32") = R.reshape(permute_dims271, R.shape([25, 14, 14, 768]))
            permute_dims272: R.Tensor((768, 768), dtype="float32") = R.permute_dims(vision_encoder_layers_9_attn_proj_weight, axes=[1, 0])
            matmul131: R.Tensor((25, 14, 14, 768), dtype="float32") = R.matmul(reshape300, permute_dims272, out_dtype="void")
            add226: R.Tensor((25, 14, 14, 768), dtype="float32") = R.add(matmul131, vision_encoder_layers_9_attn_proj_bias)
            reshape301: R.Tensor((1, 5, 5, 14, 14, 768), dtype="float32") = R.reshape(add226, R.shape([1, 5, 5, 14, 14, 768]))
            permute_dims273: R.Tensor((1, 5, 14, 5, 14, 768), dtype="float32") = R.permute_dims(reshape301, axes=[0, 1, 3, 2, 4, 5])
            reshape302: R.Tensor((1, 70, 70, 768), dtype="float32") = R.reshape(permute_dims273, R.shape([1, 70, 70, 768]))
            strided_slice16: R.Tensor((1, 64, 64, 768), dtype="float32") = R.strided_slice(reshape302, axes=[1, 2], begin=[0, 0], end=[64, 64], strides=None, assume_inbound=False)
            add227: R.Tensor((1, 64, 64, 768), dtype="float32") = R.add(add220, strided_slice16)
            layer_norm43: R.Tensor((1, 64, 64, 768), dtype="float32") = R.nn.layer_norm(add227, vision_encoder_layers_9_layer_norm2_weight, vision_encoder_layers_9_layer_norm2_bias, axes=[3], epsilon=9.9999999999999995e-07, center=True, scale=True)
            permute_dims274: R.Tensor((768, 3072), dtype="float32") = R.permute_dims(vision_encoder_layers_9_mlp_lin1_weight, axes=[1, 0])
            matmul132: R.Tensor((1, 64, 64, 3072), dtype="float32") = R.matmul(layer_norm43, permute_dims274, out_dtype="void")
            add228: R.Tensor((1, 64, 64, 3072), dtype="float32") = R.add(matmul132, vision_encoder_layers_9_mlp_lin1_bias)
            gelu21: R.Tensor((1, 64, 64, 3072), dtype="float32") = R.nn.gelu(add228)
            permute_dims275: R.Tensor((3072, 768), dtype="float32") = R.permute_dims(vision_encoder_layers_9_mlp_lin2_weight, axes=[1, 0])
            matmul133: R.Tensor((1, 64, 64, 768), dtype="float32") = R.matmul(gelu21, permute_dims275, out_dtype="void")
            add229: R.Tensor((1, 64, 64, 768), dtype="float32") = R.add(matmul133, vision_encoder_layers_9_mlp_lin2_bias)
            add230: R.Tensor((1, 64, 64, 768), dtype="float32") = R.add(add227, add229)
            layer_norm44: R.Tensor((1, 64, 64, 768), dtype="float32") = R.nn.layer_norm(add230, vision_encoder_layers_10_layer_norm1_weight, vision_encoder_layers_10_layer_norm1_bias, axes=[3], epsilon=9.9999999999999995e-07, center=True, scale=True)
            pad15: R.Tensor((1, 70, 70, 768), dtype="float32") = R.nn.pad(layer_norm44, [0, 0, 0, 6, 0, 6, 0, 0], R.const(0, "int32"),pad_mode="constant")
            reshape303: R.Tensor((1, 5, 14, 5, 14, 768), dtype="float32") = R.reshape(pad15, R.shape([1, 5, 14, 5, 14, 768]))
            permute_dims276: R.Tensor((1, 5, 5, 14, 14, 768), dtype="float32") = R.permute_dims(reshape303, axes=[0, 1, 3, 2, 4, 5])
            reshape304: R.Tensor((25, 14, 14, 768), dtype="float32") = R.reshape(permute_dims276, R.shape([25, 14, 14, 768]))
            permute_dims277: R.Tensor((768, 2304), dtype="float32") = R.permute_dims(vision_encoder_layers_10_attn_qkv_weight, axes=[1, 0])
            matmul134: R.Tensor((25, 14, 14, 2304), dtype="float32") = R.matmul(reshape304, permute_dims277, out_dtype="void")
            add231: R.Tensor((25, 14, 14, 2304), dtype="float32") = R.add(matmul134, vision_encoder_layers_10_attn_qkv_bias)
            reshape305: R.Tensor((25, 196, 3, 12, 64), dtype="float32") = R.reshape(add231, R.shape([25, 196, 3, 12, 64]))
            permute_dims278: R.Tensor((3, 25, 12, 196, 64), dtype="float32") = R.permute_dims(reshape305, axes=[2, 0, 3, 1, 4])
            reshape306: R.Tensor((3, 300, 196, 64), dtype="float32") = R.reshape(permute_dims278, R.shape([3, 300, 196, 64]))
            split22: R.Tuple(R.Tensor((1, 300, 196, 64), dtype="float32"), R.Tensor((1, 300, 196, 64), dtype="float32"), R.Tensor((1, 300, 196, 64), dtype="float32")) = R.split(reshape306, indices_or_sections=3, axis=0)
            split_022: R.Tensor((1, 300, 196, 64), dtype="float32") = split22[0]
            split_122: R.Tensor((1, 300, 196, 64), dtype="float32") = split22[1]
            split_222: R.Tensor((1, 300, 196, 64), dtype="float32") = split22[2]
            unbind66: R.Tensor((300, 196, 64), dtype="float32") = R.squeeze(split_022, axis=[0])
            unbind67: R.Tensor((300, 196, 64), dtype="float32") = R.squeeze(split_122, axis=[0])
            unbind68: R.Tensor((300, 196, 64), dtype="float32") = R.squeeze(split_222, axis=[0])
            mul115: R.Tensor((300, 196, 64), dtype="float32") = R.multiply(unbind66, R.const(0.125, "float32"))
            permute_dims279: R.Tensor((300, 64, 196), dtype="float32") = R.permute_dims(unbind67, axes=[0, 2, 1])
            matmul135: R.Tensor((300, 196, 196), dtype="float32") = R.matmul(mul115, permute_dims279, out_dtype="void")
            reshape307: R.Tensor((1, 27, 64), dtype="float32") = R.reshape(vision_encoder_layers_10_attn_rel_pos_h, R.shape([1, 27, 64]))
            permute_dims280: R.Tensor((1, 64, 27), dtype="float32") = R.permute_dims(reshape307, axes=[0, 2, 1])
            reshape308: R.Tensor((64, 27), dtype="float32") = R.reshape(permute_dims280, R.shape([64, 27]))
            permute_dims281: R.Tensor((27, 64), dtype="float32") = R.permute_dims(reshape308, axes=[1, 0])
            take44: R.Tensor((14, 14, 64), dtype="float32") = R.take(permute_dims281, add132, axis=0)
            reshape309: R.Tensor((1, 27, 64), dtype="float32") = R.reshape(vision_encoder_layers_10_attn_rel_pos_w, R.shape([1, 27, 64]))
            permute_dims282: R.Tensor((1, 64, 27), dtype="float32") = R.permute_dims(reshape309, axes=[0, 2, 1])
            reshape310: R.Tensor((64, 27), dtype="float32") = R.reshape(permute_dims282, R.shape([64, 27]))
            permute_dims283: R.Tensor((27, 64), dtype="float32") = R.permute_dims(reshape310, axes=[1, 0])
            take45: R.Tensor((14, 14, 64), dtype="float32") = R.take(permute_dims283, add132, axis=0)
            reshape311: R.Tensor((300, 14, 14, 64), dtype="float32") = R.reshape(unbind66, R.shape([300, 14, 14, 64]))
            einsum44: R.Tensor((300, 14, 14, 14), dtype="float32") = R.einsum((reshape311, take44), subscripts="bhwc,hkc->bhwk")
            einsum45: R.Tensor((300, 14, 14, 14), dtype="float32") = R.einsum((reshape311, take45), subscripts="bhwc,wkc->bhwk")
            reshape312: R.Tensor((300, 14, 14, 14, 14), dtype="float32") = R.reshape(matmul135, R.shape([300, 14, 14, 14, 14]))
            expand_dims153: R.Tensor((300, 14, 14, 1, 14), dtype="float32") = R.expand_dims(einsum44, axis=[3])
            expand_dims154: R.Tensor((300, 14, 1, 14, 14), dtype="float32") = R.expand_dims(einsum45, axis=[2])
            add234: R.Tensor((300, 14, 14, 14, 14), dtype="float32") = R.add(reshape312, expand_dims153)
            add235: R.Tensor((300, 14, 14, 14, 14), dtype="float32") = R.add(add234, expand_dims154)
            reshape313: R.Tensor((300, 196, 196), dtype="float32") = R.reshape(add235, R.shape([300, 196, 196]))
            softmax22: R.Tensor((300, 196, 196), dtype="float32") = R.nn.softmax(reshape313, axis=2)
            matmul136: R.Tensor((300, 196, 64), dtype="float32") = R.matmul(softmax22, unbind68, out_dtype="void")
            reshape314: R.Tensor((25, 12, 14, 14, 64), dtype="float32") = R.reshape(matmul136, R.shape([25, 12, 14, 14, 64]))
            permute_dims284: R.Tensor((25, 14, 14, 12, 64), dtype="float32") = R.permute_dims(reshape314, axes=[0, 2, 3, 1, 4])
            reshape315: R.Tensor((25, 14, 14, 768), dtype="float32") = R.reshape(permute_dims284, R.shape([25, 14, 14, 768]))
            permute_dims285: R.Tensor((768, 768), dtype="float32") = R.permute_dims(vision_encoder_layers_10_attn_proj_weight, axes=[1, 0])
            matmul137: R.Tensor((25, 14, 14, 768), dtype="float32") = R.matmul(reshape315, permute_dims285, out_dtype="void")
            add236: R.Tensor((25, 14, 14, 768), dtype="float32") = R.add(matmul137, vision_encoder_layers_10_attn_proj_bias)
            reshape316: R.Tensor((1, 5, 5, 14, 14, 768), dtype="float32") = R.reshape(add236, R.shape([1, 5, 5, 14, 14, 768]))
            permute_dims286: R.Tensor((1, 5, 14, 5, 14, 768), dtype="float32") = R.permute_dims(reshape316, axes=[0, 1, 3, 2, 4, 5])
            reshape317: R.Tensor((1, 70, 70, 768), dtype="float32") = R.reshape(permute_dims286, R.shape([1, 70, 70, 768]))
            strided_slice17: R.Tensor((1, 64, 64, 768), dtype="float32") = R.strided_slice(reshape317, axes=[1, 2], begin=[0, 0], end=[64, 64], strides=None, assume_inbound=False)
            add237: R.Tensor((1, 64, 64, 768), dtype="float32") = R.add(add230, strided_slice17)
            layer_norm45: R.Tensor((1, 64, 64, 768), dtype="float32") = R.nn.layer_norm(add237, vision_encoder_layers_10_layer_norm2_weight, vision_encoder_layers_10_layer_norm2_bias, axes=[3], epsilon=9.9999999999999995e-07, center=True, scale=True)
            permute_dims287: R.Tensor((768, 3072), dtype="float32") = R.permute_dims(vision_encoder_layers_10_mlp_lin1_weight, axes=[1, 0])
            matmul138: R.Tensor((1, 64, 64, 3072), dtype="float32") = R.matmul(layer_norm45, permute_dims287, out_dtype="void")
            add238: R.Tensor((1, 64, 64, 3072), dtype="float32") = R.add(matmul138, vision_encoder_layers_10_mlp_lin1_bias)
            gelu22: R.Tensor((1, 64, 64, 3072), dtype="float32") = R.nn.gelu(add238)
            permute_dims288: R.Tensor((3072, 768), dtype="float32") = R.permute_dims(vision_encoder_layers_10_mlp_lin2_weight, axes=[1, 0])
            matmul139: R.Tensor((1, 64, 64, 768), dtype="float32") = R.matmul(gelu22, permute_dims288, out_dtype="void")
            add239: R.Tensor((1, 64, 64, 768), dtype="float32") = R.add(matmul139, vision_encoder_layers_10_mlp_lin2_bias)
            add240: R.Tensor((1, 64, 64, 768), dtype="float32") = R.add(add237, add239)
            layer_norm46: R.Tensor((1, 64, 64, 768), dtype="float32") = R.nn.layer_norm(add240, vision_encoder_layers_11_layer_norm1_weight, vision_encoder_layers_11_layer_norm1_bias, axes=[3], epsilon=9.9999999999999995e-07, center=True, scale=True)
            permute_dims289: R.Tensor((768, 2304), dtype="float32") = R.permute_dims(vision_encoder_layers_11_attn_qkv_weight, axes=[1, 0])
            matmul140: R.Tensor((1, 64, 64, 2304), dtype="float32") = R.matmul(layer_norm46, permute_dims289, out_dtype="void")
            add241: R.Tensor((1, 64, 64, 2304), dtype="float32") = R.add(matmul140, vision_encoder_layers_11_attn_qkv_bias)
            reshape318: R.Tensor((1, 4096, 3, 12, 64), dtype="float32") = R.reshape(add241, R.shape([1, 4096, 3, 12, 64]))
            permute_dims290: R.Tensor((3, 1, 12, 4096, 64), dtype="float32") = R.permute_dims(reshape318, axes=[2, 0, 3, 1, 4])
            reshape319: R.Tensor((3, 12, 4096, 64), dtype="float32") = R.reshape(permute_dims290, R.shape([3, 12, 4096, 64]))
            split23: R.Tuple(R.Tensor((1, 12, 4096, 64), dtype="float32"), R.Tensor((1, 12, 4096, 64), dtype="float32"), R.Tensor((1, 12, 4096, 64), dtype="float32")) = R.split(reshape319, indices_or_sections=3, axis=0)
            split_023: R.Tensor((1, 12, 4096, 64), dtype="float32") = split23[0]
            split_123: R.Tensor((1, 12, 4096, 64), dtype="float32") = split23[1]
            split_223: R.Tensor((1, 12, 4096, 64), dtype="float32") = split23[2]
            unbind69: R.Tensor((12, 4096, 64), dtype="float32") = R.squeeze(split_023, axis=[0])
            unbind70: R.Tensor((12, 4096, 64), dtype="float32") = R.squeeze(split_123, axis=[0])
            unbind71: R.Tensor((12, 4096, 64), dtype="float32") = R.squeeze(split_223, axis=[0])
            mul120: R.Tensor((12, 4096, 64), dtype="float32") = R.multiply(unbind69, R.const(0.125, "float32"))
            permute_dims291: R.Tensor((12, 64, 4096), dtype="float32") = R.permute_dims(unbind70, axes=[0, 2, 1])
            matmul141: R.Tensor((12, 4096, 4096), dtype="float32") = R.matmul(mul120, permute_dims291, out_dtype="void")  #### CORRECT
            reshape320: R.Tensor((1, 127, 64), dtype="float32") = R.reshape(vision_encoder_layers_11_attn_rel_pos_h, R.shape([1, 127, 64]))
            permute_dims292: R.Tensor((1, 64, 127), dtype="float32") = R.permute_dims(reshape320, axes=[0, 2, 1])
            reshape321: R.Tensor((64, 127), dtype="float32") = R.reshape(permute_dims292, R.shape([64, 127]))
            permute_dims293: R.Tensor((127, 64), dtype="float32") = R.permute_dims(reshape321, axes=[1, 0])
            take46: R.Tensor((64, 64, 64), dtype="float32") = R.take(permute_dims293, add152, axis=0)
            reshape322: R.Tensor((1, 127, 64), dtype="float32") = R.reshape(vision_encoder_layers_11_attn_rel_pos_w, R.shape([1, 127, 64]))
            permute_dims294: R.Tensor((1, 64, 127), dtype="float32") = R.permute_dims(reshape322, axes=[0, 2, 1])
            reshape323: R.Tensor((64, 127), dtype="float32") = R.reshape(permute_dims294, R.shape([64, 127]))
            permute_dims295: R.Tensor((127, 64), dtype="float32") = R.permute_dims(reshape323, axes=[1, 0])
            take47: R.Tensor((64, 64, 64), dtype="float32") = R.take(permute_dims295, add152, axis=0)
            reshape324: R.Tensor((12, 64, 64, 64), dtype="float32") = R.reshape(unbind69, R.shape([12, 64, 64, 64]))
            einsum46: R.Tensor((12, 64, 64, 64), dtype="float32") = R.einsum((reshape324, take46), subscripts="bhwc,hkc->bhwk")
            einsum47: R.Tensor((12, 64, 64, 64), dtype="float32") = R.einsum((reshape324, take47), subscripts="bhwc,wkc->bhwk")
            reshape325: R.Tensor((12, 64, 64, 64, 64), dtype="float32") = R.reshape(matmul141, R.shape([12, 64, 64, 64, 64]))
            expand_dims159: R.Tensor((12, 64, 64, 1, 64), dtype="float32") = R.expand_dims(einsum46, axis=[3])
            expand_dims160: R.Tensor((12, 64, 1, 64, 64), dtype="float32") = R.expand_dims(einsum47, axis=[2])
            add244: R.Tensor((12, 64, 64, 64, 64), dtype="float32") = R.add(reshape325, expand_dims159)
            add245: R.Tensor((12, 64, 64, 64, 64), dtype="float32") = R.add(add244, expand_dims160)
            reshape326: R.Tensor((12, 4096, 4096), dtype="float32") = R.reshape(add245, R.shape([12, 4096, 4096]))
            softmax23: R.Tensor((12, 4096, 4096), dtype="float32") = R.nn.softmax(reshape326, axis=2)
            matmul142: R.Tensor((12, 4096, 64), dtype="float32") = R.matmul(softmax23, unbind71, out_dtype="void")
            reshape327: R.Tensor((1, 12, 64, 64, 64), dtype="float32") = R.reshape(matmul142, R.shape([1, 12, 64, 64, 64]))
            permute_dims296: R.Tensor((1, 64, 64, 12, 64), dtype="float32") = R.permute_dims(reshape327, axes=[0, 2, 3, 1, 4])
            reshape328: R.Tensor((1, 64, 64, 768), dtype="float32") = R.reshape(permute_dims296, R.shape([1, 64, 64, 768]))
            permute_dims297: R.Tensor((768, 768), dtype="float32") = R.permute_dims(vision_encoder_layers_11_attn_proj_weight, axes=[1, 0])
            matmul143: R.Tensor((1, 64, 64, 768), dtype="float32") = R.matmul(reshape328, permute_dims297, out_dtype="void")
            add246: R.Tensor((1, 64, 64, 768), dtype="float32") = R.add(matmul143, vision_encoder_layers_11_attn_proj_bias)
            add247: R.Tensor((1, 64, 64, 768), dtype="float32") = R.add(add240, add246)
            layer_norm47: R.Tensor((1, 64, 64, 768), dtype="float32") = R.nn.layer_norm(add247, vision_encoder_layers_11_layer_norm2_weight, vision_encoder_layers_11_layer_norm2_bias, axes=[3], epsilon=9.9999999999999995e-07, center=True, scale=True)
            permute_dims298: R.Tensor((768, 3072), dtype="float32") = R.permute_dims(vision_encoder_layers_11_mlp_lin1_weight, axes=[1, 0])
            matmul144: R.Tensor((1, 64, 64, 3072), dtype="float32") = R.matmul(layer_norm47, permute_dims298, out_dtype="void")
            add248: R.Tensor((1, 64, 64, 3072), dtype="float32") = R.add(matmul144, vision_encoder_layers_11_mlp_lin1_bias)
            gelu23: R.Tensor((1, 64, 64, 3072), dtype="float32") = R.nn.gelu(add248)
            permute_dims299: R.Tensor((3072, 768), dtype="float32") = R.permute_dims(vision_encoder_layers_11_mlp_lin2_weight, axes=[1, 0]) 
            matmul145: R.Tensor((1, 64, 64, 768), dtype="float32") = R.matmul(gelu23, permute_dims299, out_dtype="void") #### matmul145 correct
            add249: R.Tensor((1, 64, 64, 768), dtype="float32") = R.add(matmul145, vision_encoder_layers_11_mlp_lin2_bias)
            add250: R.Tensor((1, 64, 64, 768), dtype="float32") = R.add(add247, add249)
            lv4: R.Tensor((256, 1, 1, 768), dtype="float32") = R.permute_dims(vision_encoder_neck_conv1_weight, axes=[0, 2, 3, 1])
            conv2d4: R.Tensor((1, 64, 64, 256), dtype="float32") = R.nn.conv2d(add250, lv4, strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NHWC", kernel_layout="OHWI", out_layout="NHWC", out_dtype="void")
            mean8: R.Tensor((1, 64, 64, 1), dtype="float32") = R.mean(conv2d4, axis=[3], keepdims=True)
            subtract54: R.Tensor((1, 64, 64, 256), dtype="float32") = R.subtract(conv2d4, mean8)
            mean9: R.Tensor((1, 64, 64, 256), dtype="float32") = R.power(subtract54, R.const(2, "float32"))
            mean10: R.Tensor((1, 64, 64, 1), dtype="float32") = R.mean(mean9, axis=[3], keepdims=True)
            add251: R.Tensor((1, 64, 64, 1), dtype="float32") = R.add(mean10, R.const(9.9999999747524271e-07, "float32"))
            mean11: R.Tensor((1, 64, 64, 1), dtype="float32") = R.sqrt(add251)
            divide6: R.Tensor((1, 64, 64, 256), dtype="float32") = R.divide(subtract54, mean11)
            expand_dims161: R.Tensor((256, 1), dtype="float32") = R.expand_dims(vision_encoder_neck_layer_norm1_weight, axis=[1])
            expand_dims162: R.Tensor((256, 1, 1), dtype="float32") = R.expand_dims(expand_dims161, axis=[1])
            expand_dims163: R.Tensor((256, 1), dtype="float32") = R.expand_dims(vision_encoder_neck_layer_norm1_bias, axis=[1])
            expand_dims164: R.Tensor((256, 1, 1), dtype="float32") = R.expand_dims(expand_dims163, axis=[1])
            lv5_1: R.Tensor((1, 256, 64, 64), dtype="float32") = R.permute_dims(divide6, axes=[0, 3, 1, 2])
            mul125: R.Tensor((1, 256, 64, 64), dtype="float32") = R.multiply(expand_dims162, lv5_1)
            add252: R.Tensor((1, 256, 64, 64), dtype="float32") = R.add(mul125, expand_dims164)
            lv6_1: R.Tensor((1, 64, 64, 256), dtype="float32") = R.permute_dims(add252, axes=[0, 2, 3, 1])
            lv7: R.Tensor((256, 3, 3, 256), dtype="float32") = R.permute_dims(vision_encoder_neck_conv2_weight, axes=[0, 2, 3, 1])
            conv2d5: R.Tensor((1, 64, 64, 256), dtype="float32") = R.nn.conv2d(lv6_1, lv7, strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=1, data_layout="NHWC", kernel_layout="OHWI", out_layout="NHWC", out_dtype="void")
            mean12: R.Tensor((1, 64, 64, 1), dtype="float32") = R.mean(conv2d5, axis=[3], keepdims=True)
            subtract55: R.Tensor((1, 64, 64, 256), dtype="float32") = R.subtract(conv2d5, mean12)
            mean13: R.Tensor((1, 64, 64, 256), dtype="float32") = R.power(subtract55, R.const(2, "float32"))
            mean14: R.Tensor((1, 64, 64, 1), dtype="float32") = R.mean(mean13, axis=[3], keepdims=True)
            add253: R.Tensor((1, 64, 64, 1), dtype="float32") = R.add(mean14, R.const(9.9999999747524271e-07, "float32"))
            mean15: R.Tensor((1, 64, 64, 1), dtype="float32") = R.sqrt(add253)
            divide7: R.Tensor((1, 64, 64, 256), dtype="float32") = R.divide(subtract55, mean15)
            expand_dims165: R.Tensor((256, 1), dtype="float32") = R.expand_dims(vision_encoder_neck_layer_norm2_weight, axis=[1])
            expand_dims166: R.Tensor((256, 1, 1), dtype="float32") = R.expand_dims(expand_dims165, axis=[1])
            expand_dims167: R.Tensor((256, 1), dtype="float32") = R.expand_dims(vision_encoder_neck_layer_norm2_bias, axis=[1])
            expand_dims168: R.Tensor((256, 1, 1), dtype="float32") = R.expand_dims(expand_dims167, axis=[1])
            lv8: R.Tensor((1, 256, 64, 64), dtype="float32") = R.permute_dims(divide7, axes=[0, 3, 1, 2])
            mul126: R.Tensor((1, 256, 64, 64), dtype="float32") = R.multiply(expand_dims166, lv8)
            add254: R.Tensor((1, 256, 64, 64), dtype="float32") = R.add(mul126, expand_dims168)
            add255: R.Tensor((1, 1, 1, 2), dtype="float32") = R.add(input_points, R.const(0.5, "float32"))
            concat6: R.Tensor((1, 1, 2, 2), dtype="float32") = R.concat((add255, metadata["relax.expr.Constant"][6]), axis=2)
            strided_slice18: R.Tensor((1, 1, 2, 1), dtype="float32") = R.strided_slice(concat6, axes=[3], begin=[0], end=[1], strides=None, assume_inbound=False)
            divide8: R.Tensor((1, 1, 2, 1), dtype="float32") = R.divide(strided_slice18, R.const(1024, "float32"))
            strided_slice19: R.Tensor((1, 1, 2, 1), dtype="float32") = R.strided_slice(concat6, axes=[3], begin=[1], end=[2], strides=None, assume_inbound=False)
            divide9: R.Tensor((1, 1, 2, 1), dtype="float32") = R.divide(strided_slice19, R.const(1024, "float32"))
            concat8: R.Tensor((1, 1, 2, 2), dtype="float32") = R.concat((divide8, divide9), axis=3)
            add256: R.Tensor((1, 1, 2, 2), dtype="float32") = R.add(concat8, concat8)
            subtract56: R.Tensor((1, 1, 2, 2), dtype="float32") = R.subtract(add256, R.const(1, "float32"))
            matmul146: R.Tensor((1, 1, 2, 128), dtype="float32") = R.matmul(subtract56, prompt_encoder_shared_embedding_positional_embedding, out_dtype="void")
            #### Correct
            mul128: R.Tensor((1, 1, 2, 128), dtype="float32") = R.multiply(R.const(6.2831854820251465, "float32"), matmul146)
            ############## Correct
            # p1 = R.print(mul128, format="x: {}")
            # _io1: R.Object = R.call_pure_packed("effect.print", _io, gelu, sinfo_args=(R.Object,))
            # _io1: R.Object = R.call_pure_packed("effect.print", _io, mul128, sinfo_args=(R.Tensor((1, 1, 2, 128), dtype="float32")))
            sin2: R.Tensor((1, 1, 2, 128), dtype="float32") = R.sin(mul128)
            ##### Error
            cos2: R.Tensor((1, 1, 2, 128), dtype="float32") = R.cos(mul128)
            #### Error
            mul_add = R.add(mul128, R.const(1, "float32"))
            # mul_sub = R.subtract(mul_add, mul128)
            # yongwww = R.cos(mul_add)# R.nn.gelu(mul128)# R.add(mul128, mul128)-> correct with [add, gelu, sqrt, square, tanh, sigmoid]. Wrong with [sin, cos, ]
            # yongwww = cos2
            R.output(cos2)
            # R.output(cos2, mul128)
        return cos2 # correct: [matmul141, matmul145, matmul146, lv_1], wrong: [add257, matmul161, lv1_1: second last row, matmul150: wrong on the second column]
        # concat11: second last column wrong

        ##### Debugging ideas #####
        # o0: compare R.cos(mul128) with R.square, looking in to the cutlass subgraph, and the IR after RunCodegen
        #     names for failure: sam_cos_cutlass_matmul_partition.py, sam_cos_cutlass_matmul_runcodegen.py, sam_cos_cutlass_matmul_ex.py
        #     names for success: sam_square_cutlass_matmul_partition.py, sam_square_cutlass_matmul_runcodegen.py, sam_square_cutlass_matmul_ex.py

        ### Campare executables : Looks fine
        ### Compare Logs in the RunCodegen: Likely the fused_relax_matmul_cutlass was generated with wrong probably: Logs looks fine
        #      sam_cos_cutlass_matmul_runcodegen.log, sam_square_cutlass_matmul_runcodegen.log
        ### Just leave main: Error out as well
        ### Update R.func_attr({"global_symbol": "forward", "num_input": 3}) # Error out as well

        ### Proposed checks
        # 1. compare sin and cos with square: Looks fine
        # 2. compare the logs beween cos with sqrt, because sqrt are using same unary common code

        
        ####  Try to get a minimal test case: failed



def get_inputs():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
    input_points = [[[450, 600]]]  # 2D location of a window in the image

    inputs = processor(raw_image, input_points=input_points, return_tensors="pt").to(device)
    return inputs


def get_transformers_torch_sam(type="base"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if type == "base":
        return PTSamModel(SamConfig()).to(device)
        tmp_sam = PTSamModel.from_pretrained("facebook/sam-vit-base") #.to(device)
    else:
        tmp_sam = PTSamModel.from_pretrained("facebook/sam-vit-huge")
    
    config = tmp_sam.config
    return PTSamModel(config).to(device)
    # sam_huge_config = model.config

def _offload_to_cutlass(mod, target, entry_functions=["main", "get_prompt_embeddings", "get_image_embeddings"]):
    # Currently, sm86 is not supported.
    sm = int(target.arch.split("_")[1])
    print("sm: ", sm)
    if sm > 80:
        sm = 80
    mod = partition_for_cutlass(mod)
    
    # print("Module with R.square after cutlass partition: \n", mod.script(show_meta=True))
    # mod.show()
    mod = relax.transform.RunCodegen(
        {"cutlass": {"sm": sm, "find_first_valid": False}},
        entry_functions=entry_functions,
    )(mod)
    # print("Module with R.square after cutlass RunCodegen: \n", mod.script(show_meta=True))

    return mod

def test_matmul_cutlass(sam_model=None, apply_cutlass=True):
    entry_name = "main"

    # target, dev = "llvm", tvm.cpu()
    target = tvm.target.Target("nvidia/nvidia-a100")  # tvm.target.Target("cuda", host="llvm")
    dev = tvm.gpu()
    ir_mod = SAMModuleDebug
    mod = tvm.ir.IRModule()
    mod["main"] = ir_mod["main"]
    # mod["get_prompt_embeddings"] = ir_mod["get_prompt_embeddings"]
    # mod["get_image_embeddings"] = ir_mod["get_image_embeddings"]

    mod["_initialize_effect"] = ir_mod["_initialize_effect"]

    mod = run_opt_passes(mod, combine_matmul=False)
    # print(mod.script(show_meta=True))
    if apply_cutlass:
        mod = _offload_to_cutlass(mod, target, ["main"])
    
    # print(mod.script(show_meta=True))

    mod = run_lower_passes(mod, target, do_tuning=False)

    # print(mod.script(show_meta=True))
    exe = relax.build(mod, target=target)
    # print("Sam with cos exe: \n", exe.as_text())
    vm = relax.VirtualMachine(exe, dev)

    # Prepare inputs for inference
    tvm_params = {}
    for k, v in sam_model.state_dict().items():
        tvm_params[k.replace(".", "_")] = tvm.nd.array(v.cpu().numpy(), dev)

    # image input
    img_inputs = get_inputs()
    for k, v in img_inputs.items():
        tvm_params[k] = tvm.nd.array(v.cpu().float().numpy(), dev)

    effects = vm["_initialize_effect"]()
    tvm_params[".io"] = effects
    tvm_params["_io"] = effects
    tvm_params["._io"] = effects

    # Convert param into ordered list.
    func_arity = vm._get_function_arity(entry_name)
    tvm_names = [vm._get_function_param_name(entry_name, i) for i in range(func_arity)]
    # print("tvm_names: ", tvm_names)
    # print("tvm_params names: ", tvm_params.keys())

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

    out_np = _to_numpy(out_nd)

    # print("Relax SAM inference output: ", out_np)#[0][0])
    return out_np#[0][0]


if __name__ == "__main__" :
    pt_sam_model = get_transformers_torch_sam("base")
    out1 = test_matmul_cutlass(pt_sam_model, apply_cutlass=True)
    out2 = test_matmul_cutlass(pt_sam_model, apply_cutlass=False)

    tvm.testing.assert_allclose(out1, out2, rtol=1e-1, atol=1e-1)
    print("Inference results match!")
