
import torch
import torch.nn as nn
import math
from jaxtyping import Float, Int, Bool
from torch import Tensor
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.swiglu import SwiGLU
from cs336_basics.mha import MultiheadSelfAttention
from cs336_basics.sdpa import sdpa

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int, 
        pos_layer: nn.Module | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Create a causal multihead self-attention layer.

        Args:
            d_model (int): Positive model dimension, divisible by num_heads.
            num_heads (int): Positive number of attention heads.
            d_ff (int): Intermediate dimension.
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

        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiheadSelfAttention(d_model, num_heads,
                                 pos_layer, 
                                 device, 
                                 dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)


    def forward(
        self,
        x: Float[Tensor, "... seq_len d_model"],
        token_positions: Int[Tensor, "... seq_len"] | None = None,
        mask: Bool[Tensor, "..."] | None = None,
    ) -> Float[Tensor, "... seq_len d_model"]:
        """Apply transformer block to the input.

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
        x += self.attn(self.ln1(x), token_positions, mask)
        x += self.ffn(self.ln2(x))

        return x
        
