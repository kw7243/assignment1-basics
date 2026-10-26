import torch
import torch.nn as nn
import math
from jaxtyping import Float, Int
from torch import Tensor
from cs336_basics.linear import Linear

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None) -> None:
        """
        Create RoPE layer

        Args:
            theta: float  Θ value for the RoPE
            d_k: int  dimension of query and key vectors
            max_seq_len: int  Maximum sequence length that will be input
            device: torch.device | None = None  Device to store the buffer on
        """
        super().__init__()

        assert d_k > 0 and d_k % 2 == 0, "Dimensions not positive even number"

        # Module params
        dims = torch.arange(0, d_k, 2, device=device)

        # unsqueeze(1) = [:, None]
        freqs = (theta ** (-dims/d_k)).unsqueeze(1) # (d_k/2,1) float32

        positions = torch.arange(max_seq_len,  dtype=freqs.dtype)[:, None] # (L_max,1)

        # matmul reqs matching dtypes, but broadcasting upgrades int --> float
        angles = positions @ freqs.T # (L_max, d_k/2) float32

        self.register_buffer(
            "cos_table", angles.cos(), persistent=False
        )
        self.register_buffer(
            "sin_table", angles.sin(), persistent=False
        )


    def forward(self, x: Float[Tensor, " ... seq_len d_k"], 
                token_positions: Int[Tensor, " ... seq_len"]) -> Float[Tensor, " ... seq_len d_k"]:
        """
        Args:
            x (Int[Tensor, "... seq_len d_k"]): input tensor (ex. (B,L,D))
            token_positions (Int[Tensor, " ... seq_len"]): token positions for a batch 
                                            (diff reqs process seqs at different positions)

        Returns:
            Float[Tensor, "... seq_len d_k"]: RoPE(x)
                                            ex. (B,L,D)
        """

        x_even = x[..., 0::2] # (B,L,D/2)
        x_odd = x[..., 1::2]

        cos = self.cos_table[token_positions].to(x.dtype) # (B,L,D/2) x.dtype
        sin = self.sin_table[token_positions].to(x.dtype) # (B,L,D/2) x.dtype

        y_even = cos*x_even  - sin*x_odd # (B,L,D/2) x.dtype x (B,L,D/2)
        y_odd = sin*x_even + cos*x_odd

        return torch.stack((y_even, y_odd), dim=-1).flatten(-2) # (B,L,D/2) --> (B,L,D/2,2) --> (B,L,D) x.dtype











    

