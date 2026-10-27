import torch
import torch.nn as nn
import math
from jaxtyping import Float, Int, Bool
from torch import Tensor
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

    

    



