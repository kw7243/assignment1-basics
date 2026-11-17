import math

import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int
from torch import Tensor

from cs336_basics.linear import Linear
from cs336_basics.softmax import softmax
from cs336_basics.kv_cache import KVCache

def sdpa(
    q: Float[Tensor, "... query_len d_k"],
    k: Float[Tensor, "... key_len d_k"],
    v: Float[Tensor, "... key_len d_v"],
    mask: Bool[Tensor, "..."] | None = None,
) -> Float[Tensor, "... query_len d_v"]:
    """Compute scaled dot-product attention.

    Args:
        q (Float[Tensor, "... query_len d_k"]): Queries.
        k (Float[Tensor, "... key_len d_k"]): Keys.
        v (Float[Tensor, "... key_len d_v"]): Values to aggregate.
        mask (Bool[Tensor, "..."] | None): Broadcasts to the attention scores
            (..., query_len, key_len). True allows a key position.

    Returns:
        Float[Tensor, "... query_len d_v"]: Attention-weighted values. Leading
            dimensions follow input broadcasting.

    Requires:
        q, k, and v have matching floating-point dtypes and devices.
        Their leading dimensions are broadcastable, and d_k is positive.
        mask, when supplied, is on the same device as the inputs.
        Each query has at least one allowed key.
    """
    d_k = q.shape[-1]
    attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k) # (..., query_len, key_len)

    if mask is not None:
        attn_scores = attn_scores.masked_fill(~mask, float("-inf")) 

    attn_probs = softmax(attn_scores, dim=-1) 

    return attn_probs @ v # (B,L,D)


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        pos_layer: nn.Module | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        *,
        layer_idx: int | None = None,
    ) -> None:
        """Create a causal multihead self-attention layer.

        Args:
            d_model (int): Positive model dimension, divisible by num_heads.
            num_heads (int): Positive number of attention heads.
            pos_layer (nn.Module | None): Optional positional layer for queries and keys. It must
                accept projected tensors and token positions.
            layer_idx (int | None): Fixed layer index in the model. Required
                when a forward call supplies a KV cache; otherwise optional.
            device (torch.device | str | None): Device for the projection parameters.
            dtype (torch.dtype | None): Floating-point dtype for the projection parameters.

        Requires:
            pos_layer, when supplied, is compatible with the projection device
            and dtype. These arguments do not move or cast pos_layer.

        Raises:
            ValueError: If either dimension is not positive, or d_model is not
                divisible by num_heads.
        """
        super().__init__()

        if d_model <= 0 or num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError("d_model and num_heads must be positive, and d_model must be divisible by num_heads.")

        # Module params
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_head = self.d_model // num_heads
        self.layer_idx = layer_idx

        self.q_proj = Linear(self.d_model, self.d_model, device, dtype)
        self.k_proj = Linear(self.d_model, self.d_model, device, dtype)
        self.v_proj = Linear(self.d_model, self.d_model, device, dtype)
        self.output_proj = Linear(self.d_model, self.d_model, device, dtype)

        self.pos_layer = pos_layer


    def forward(
        self,
        x: Float[Tensor, "... seq_len d_model"],
        token_positions: Int[Tensor, "... seq_len"] | None = None,
        mask: Bool[Tensor, "..."] | None = None,
        kv_cache: KVCache | None = None
    ) -> Float[Tensor, "... seq_len d_model"]:
        """Apply causal self-attention with an optional external KV cache.

        Args:
            x (Float[Tensor, "... seq_len d_model"]): Sequence representations.
            token_positions (Int[Tensor, "... seq_len"] | None): Optional
                absolute positions. Defaults to the cached prefix length plus
                local positions, or local positions when no cache is supplied.
                Unused when there is no positional layer.
            mask (Bool[Tensor, "..."] | None): Broadcasts to the attention scores
                (..., num_heads, L, T + L), where T is the cached length and L
                is the number of new tokens. True allows a key position.
                The causal mask also applies.
            kv_cache (KVCache | None): Externally owned cache for this call.
                Requires a layer_idx assigned at construction. The module
                does not retain the cache between calls.

        Returns:
            Float[Tensor, "... seq_len d_model"]: Contextualized sequence representations.

        Requires:
            x is compatible with the projection parameter dtype and device.
            mask, when supplied, is on the same device as x.
            token_positions meets the positional layer's shape, index, and
            device requirements. Each query has at least one allowed key.
            Cached execution uses a (B, L, d_model) input and a contiguous
            cached prefix of length T. Explicit cached token positions are
            T, ..., T + L - 1.

        Raises:
            ValueError: If the input width does not match d_model, or a cache
                is supplied without a layer_idx assigned at construction.
        """
        *batch_dims, L, d_model = x.shape
        if d_model != self.d_model:
            raise ValueError(f"Expected input width {self.d_model}, got {d_model}.")

        # Cached prefix length before appending new tokens.
        layer_idx = self.layer_idx
        T = 0
        if kv_cache is not None:
            if layer_idx is None:
                raise ValueError("layer_idx must be set when using a KV cache.")
            T = kv_cache.get_length(layer_idx)

        # Q,K,V proj
        q = self.q_proj(x) # (B,L,D)x(D,D)
        k = self.k_proj(x) 
        v = self.v_proj(x)

        q = q.reshape(*batch_dims, L, self.num_heads, self.d_head).transpose(-2,-3) # (B,H,L,d)
        k = k.reshape(*batch_dims, L, self.num_heads, self.d_head).transpose(-2,-3) # (B,H,L,d)
        v = v.reshape(*batch_dims, L, self.num_heads, self.d_head).transpose(-2,-3) # (B,H,L,d)

        # apply positional encodings
        if self.pos_layer is not None:
            if token_positions is None: # positions [T,...,T+L-1]; T=0 for prefill
                token_positions = (T + torch.arange(L, device=x.device)).expand(*batch_dims, L) # (B,L)

            tok_pos = token_positions.unsqueeze(-2) # (B,L) --> (B,1,L)
            q, k = self.pos_layer(q, tok_pos), self.pos_layer(k, tok_pos)

        if kv_cache is not None: # update KV cache
            assert layer_idx is not None
            kv_cache.append(layer_idx, k, v)
            k, v = kv_cache.get(layer_idx) # (B,H,T+L,d)

        # masking
        q_indices = T + torch.arange(L, device=x.device) # (L,), [T,...,T+L-1]
        k_indices = torch.arange(T + L, device=x.device) # (T+L,), [0,...,T+L-1]
        attn_mask = k_indices[None, :] <= q_indices[:, None] # (L,T+L); causal when T=0 too

        if mask is not None:
            attn_mask = attn_mask & mask

        # sdpa
        mh_sdpa = sdpa(q, k, v, attn_mask) # (B,H,L,d)x(B,H,d,T+L)x(B,H,T+L,d) = (B,H,L,d)
        mh_sdpa = mh_sdpa.transpose(-2,-3).reshape(*batch_dims, L, self.d_model) # (B,L,H*d)=(B,L,D)

        return self.output_proj(mh_sdpa) # (B,L,D)x(D,D)
