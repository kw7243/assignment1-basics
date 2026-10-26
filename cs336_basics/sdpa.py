import torch
import torch.nn as nn
import math
from jaxtyping import Float, Int, Bool
from torch import Tensor
from cs336_basics.softmax import softmax

def sdpa(q: Float | Int [Tensor, "batch_size ... seq_len d_k"],
        k: Float | Int [Tensor, "batch_size ... seq_len d_k"],
        v: Float | Int [Tensor, "batch_size ... seq_len d_v"], 
        mask: Bool [Tensor, "seq_len seq_len"] | None = None) -> Float | Int [Tensor, "batch_size ... seq_len d_v"]:
    """
    Args:
        x (Int[Tensor, "... d_model"]): input tensor (ex. (B,L,D))

    Returns:
        Float | Int [Tensor, "batch_size ... seq_len d_v"]: 
    """
    d_k = q.shape[-1]
    attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k) # (B,L,L)

    if mask is not None:
        attn_scores = attn_scores.masked_fill(~mask, float("-inf")) 

    attn_probs = softmax(attn_scores, dim=-1) 

    return attn_probs @ v # (B,L,D)

    

    



