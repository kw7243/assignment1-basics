import math

import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int
from torch import Tensor

from cs336_basics.linear import Linear
from cs336_basics.softmax import softmax


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
    attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k) # (B,L,L)

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
    ) -> None:
        """Create a causal multihead self-attention layer.

        Args:
            d_model (int): Positive model dimension, divisible by num_heads.
            num_heads (int): Positive number of attention heads.
            pos_layer (nn.Module | None): Optional positional layer for queries and keys. It must
                accept projected tensors and token positions.
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
    ) -> Float[Tensor, "... seq_len d_model"]:
        """Apply causal self-attention to the input.

        Args:
            x (Float[Tensor, "... seq_len d_model"]): Sequence representations.
            token_positions (Int[Tensor, "... seq_len"] | None): Required when
                pos_layer is supplied; otherwise unused.
            mask (Bool[Tensor, "..."] | None): Broadcasts to the attention scores
                (..., num_heads, seq_len, seq_len). True allows a key position.
                The causal mask also applies.

        Returns:
            Float[Tensor, "... seq_len d_model"]: Contextualized sequence representations.

        Requires:
            x is compatible with the projection parameter dtype and device.
            mask, when supplied, is on the same device as x.
            token_positions meets the positional layer's shape, index, and
            device requirements. Each query has at least one allowed key.

        Raises:
            ValueError: If the input width does not match d_model, or positions
                are missing when a positional layer is supplied.
        """
        *batch_dims, seq_len, d_model = x.shape
        if d_model != self.d_model:
            raise ValueError(f"Expected input width {self.d_model}, got {d_model}.")

        # Q,K,V proj
        q = self.q_proj(x) # (B,L,D)x(D,D)
        k = self.k_proj(x) 
        v = self.v_proj(x)

        q = q.reshape(*batch_dims, seq_len, self.num_heads, self.d_head).transpose(-2,-3) # (B,H,L,d)
        k = k.reshape(*batch_dims, seq_len, self.num_heads, self.d_head).transpose(-2,-3) # (B,H,L,d)
        v = v.reshape(*batch_dims, seq_len, self.num_heads, self.d_head).transpose(-2,-3) # (B,H,L,d)

        # apply positional encodings
        if self.pos_layer is not None:
            if token_positions is None: # default to [0,..,L-1] for prefill
                token_positions = torch.arange(0, seq_len).expand(*batch_dims, seq_len) # (B,L)
            tok_pos = token_positions.unsqueeze(-2) # (B,L) --> (B,1,L)
            q, k = self.pos_layer(q, tok_pos), self.pos_layer(k, tok_pos)

        # masking
        attn_mask = torch.ones(
            seq_len, seq_len,
            dtype=torch.bool,
            device=x.device,
        ).tril() # causal mask default

        if mask is not None:
            attn_mask = attn_mask & mask

        # sdpa
        mh_sdpa = sdpa(q, k, v, attn_mask) # (B,H,L,d)x(B,H,d,L)x(B,H,L,d) = (B,H,L,d)
        mh_sdpa = mh_sdpa.transpose(-2,-3).reshape(*batch_dims, seq_len, self.d_model) # (B,L,H*d)=(B,L,D)

        return self.output_proj(mh_sdpa) # (B,L,D)x(D,D)
