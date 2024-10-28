# shape of weight of nn linear in MLP of Llama-3.1 8B: Tensor([28672, 4096], "float32")

import pytest

import tvm
import tvm.testing
import numpy as np

from tvm import relax, tir
from tvm.ir import assert_structural_equal
import mlc_llm
from tvm.relax.frontend import nn
from mlc_llm.nn import PagedKVCache
from tvm.script import ir as I, relax as R, tir as T


def compile(mod
) -> relax.VirtualMachine:
    # compile the model
    mod = relax.transform.LegalizeOps()(mod)
    tgt = tvm.target.Target("nvidia/nvidia-h100")
    with tgt:
        mod = tvm.tir.transform.DefaultGPUSchedule()(mod)
    ex = relax.build(mod, tgt)
    return relax.VirtualMachine(ex, tvm.gpu())


def test_llama_3_1_mlp():

    class LlamaMLP(nn.Module):
        def __init__(self, hidden_size: int, intermediate_size: int):
            super().__init__()
            self.gate_proj = nn.Linear(
                in_features=hidden_size,
                out_features=intermediate_size,
                dtype="float16",
                bias=False,
            )
            self.up_proj = nn.Linear(
                in_features=hidden_size,
                out_features=intermediate_size,
                dtype="float16",
                bias=False,
            )
            self.down_proj = nn.Linear(
                intermediate_size,
                hidden_size,
                dtype="float16",
                bias=False,
            )
        def forward(self, x: nn.Tensor):
            gate = self.gate_proj(x)
            up = self.up_proj(x)
            return self.down_proj(nn.op.silu(gate) * up)
    
    class LlamaFFN(nn.Module):
        def __init__(self, hidden_size: int, intermediate_size: int, tensor_parallel_shards: int = 1):
            super().__init__()
            intermediate_size = intermediate_size // tensor_parallel_shards
            self.gate_up_proj = nn.Linear(
                in_features=hidden_size,
                out_features=2 * intermediate_size,
                bias=False,
                dtype="float16",
            )
            self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, dtype="float16")
        
        def forward(self, x: nn.Tensor):
            concat_x1_x2 = self.gate_up_proj(x)
            x1, x2 = nn.op.split(concat_x1_x2, 2, axis=-1)
            return self.down_proj(nn.op.silu(x1) * x2)

        

    hidden_size = 4096
    intermediate_size = 28672
    slm_mod = LlamaFFN(hidden_size=hidden_size, intermediate_size=intermediate_size)
    exported_mod, _ = slm_mod.export_tvm(
        spec={
            "forward": {
                "x": nn.spec.Tensor((tir.Var("seq_len", "int64"), hidden_size), "float16")
            },
        },
        debug=False,
    )

    exported_mod.show()
    vm = compile(exported_mod)
    np_ipt = np.random.rand(2, hidden_size).astype(np.float16)
    ipt = tvm.nd.array(np_ipt, tvm.cuda())
    gate_proj_weight = tvm.nd.array(np.random.rand(intermediate_size, hidden_size).astype(np.float16), tvm.cuda())
    up_proj_weight = tvm.nd.array(np.random.rand(intermediate_size, hidden_size).astype(np.float16), tvm.cuda())
    down_proj_weight = tvm.nd.array(np.random.rand(hidden_size, intermediate_size).astype(np.float16), tvm.cuda())
    res = vm["forward"](ipt, gate_proj_weight, up_proj_weight, down_proj_weight)



    def lm_head_forward(self, x: nn.Tensor):
        """The lm_head forwarding, which transposes the weight and multiplies
        with the input tensor.
        """
        weight = nn.op.permute_dims(self.weight)
        return nn.op.matmul(x, weight, out_dtype="float32")
    
# "tie_word_embeddings": false,
# Need to figure out the rule for different tp, different batch, etc.
# get_logits


# The shape of input for prefill is R.Tensor((1, "seq_len", 4096), for seq_len we can use 300, 2048, 30k

num_hidden_layers = 32 # todo (yongwww): get from config.num_hidden_layers

def test_llama_attention():
    class LlamaAttention(nn.Module):  # pylint: disable=too-many-instance-attributes
        def __init__(self, head_dim: int, hidden_size: int, num_attention_heads: int, num_key_value_heads: int, tensor_parallel_shards: int = 1):
            self.head_dim = head_dim
            self.num_q_heads = num_attention_heads // tensor_parallel_shards
            self.num_kv_heads = num_key_value_heads // tensor_parallel_shards
            self.qkv_proj = nn.Linear(
                in_features=hidden_size,
                out_features=(self.num_q_heads + 2 * self.num_kv_heads) * self.head_dim,
                bias=False,
                dtype="float16",
            )
            self.o_proj = nn.Linear(self.num_q_heads * self.head_dim, hidden_size, bias=False, dtype="float16")

        def forward(self, hidden_states: nn.Tensor, paged_kv_cache: PagedKVCache, layer_id: int):
            d, h_q, h_kv = self.head_dim, self.num_q_heads, self.num_kv_heads
            b, s, _ = hidden_states.shape
            # QKV Projection
            qkv = self.qkv_proj(hidden_states)
            qkv = nn.op.reshape(qkv, (b, s, h_q + h_kv + h_kv, d))
            # Attention
            output = nn.op.reshape(
                paged_kv_cache.attention_with_fused_qkv(layer_id, qkv, self.num_q_heads),
                (b, s, h_q * d),
            )
            return self.o_proj(output)
    
    
    head_dim = 128
    hidden_size = 4096
    num_attention_heads = 32
    num_key_value_heads = 32
    tensor_parallel_shards = 1
    slm_mod = LlamaAttention(head_dim=head_dim, 
                             hidden_size=hidden_size,
                             num_attention_heads=num_attention_heads,
                             num_key_value_heads=num_key_value_heads,
                             tensor_parallel_shards=tensor_parallel_shards)
    exported_mod, _ = slm_mod.export_tvm(
        spec={
            "forward": {
                "hidden_states": nn.spec.Tensor((1, tir.Var("seq_len", "int64"), hidden_size), "float16"),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "layer_id": nn.spec.Int,
            },
        },
        debug=False,
    )

    exported_mod.show()

if __name__ == "__main__":

    # test_llama_3_1_mlp()
    test_llama_attention()