"""Construct public model components for the cache integration tests.

Keep imports local so incomplete production code does not prevent collection
of the new test module. Existing adapters and tests are left unchanged.
"""

import torch


def get_transformer_lm(config, weights):
    from cs336_basics.transformer_lm import TransformerLM

    reference = weights["lm_head.weight"]
    model = TransformerLM(**config, device=reference.device, dtype=reference.dtype)
    model.load_state_dict(weights)
    return model.eval()


def get_transformer_block(config, weights, layer_idx):
    from cs336_basics.rope import RotaryPositionalEmbedding
    from cs336_basics.transformer import TransformerBlock

    reference = weights["ln1.weight"]
    rope = RotaryPositionalEmbedding(
        config["rope_theta"], config["d_model"] // config["num_heads"], config["context_length"], reference.device
    )
    block = TransformerBlock(
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        pos_layer=rope,
        layer_idx=layer_idx,
        device=reference.device,
        dtype=reference.dtype,
    )
    block.load_state_dict(weights)
    return block.eval()


def get_self_attention(config, weights, layer_idx, use_rope=True):
    from cs336_basics.attention import MultiHeadSelfAttention
    from cs336_basics.rope import RotaryPositionalEmbedding

    reference = weights["q_proj.weight"]
    rope = None
    if use_rope:
        rope = RotaryPositionalEmbedding(
            config["rope_theta"], config["d_model"] // config["num_heads"], config["context_length"], reference.device
        )
    attention = MultiHeadSelfAttention(
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        pos_layer=rope,
        layer_idx=layer_idx,
        device=reference.device,
        dtype=reference.dtype,
    )
    attention.load_state_dict(weights)
    return attention.eval()


def get_integration_cache(config, batch_size):
    from cs336_basics.kv_cache import KVCache

    return KVCache(
        num_layers=config["num_layers"],
        batch_size=batch_size,
        num_kv_heads=config["num_heads"],
        max_seq_len=config["context_length"],
        d_head=config["d_model"] // config["num_heads"],
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
