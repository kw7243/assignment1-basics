import torch
import torch.nn as nn
import math
from jaxtyping import Float, Int
from torch import Tensor
from cs336_basics.linear import Linear

class RotaryPositionalEmbedding(nn.Module):
    cos_table: Tensor
    sin_table: Tensor

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | str | None = None,
    ) -> None:
        """
        Create a rotary positional embedding layer.

        Args:
            theta (float): Base for the rotary frequencies.
            d_k (int): Positive, even dimension of each query or key vector.
            max_seq_len (int): Number of token positions to cache.
            device (torch.device | str | None): Device for the cached sine and cosine buffers.

        The buffers use the default floating-point dtype at construction.
        Forward converts selected values to the input dtype.

        Raises:
            ValueError: If d_k is not positive and even.
        """
        super().__init__()

        if d_k <= 0 or d_k % 2 != 0:
            raise ValueError("d_k must be a positive even number.")

        # Module params
        dims = torch.arange(0, d_k, 2, device=device)

        # unsqueeze(1) = [:, None]
        freqs = (theta ** (-dims/d_k)).unsqueeze(1) # (d_k/2,1) float32

        positions = torch.arange(max_seq_len, device=freqs.device, dtype=freqs.dtype)[:, None] # (L_max,1)

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
        Rotate vectors at the supplied token positions.

        Args:
            x (Float[Tensor, "... seq_len d_k"]): Vectors to rotate.
            token_positions (Int[Tensor, "... seq_len"]): Leading dimensions
                must broadcast against those of x.

        Returns:
            Float[Tensor, "... seq_len d_k"]: Preserves the input dtype and device.

        Requires:
            x and the module buffers are on the same device.
            token_positions contains valid indexing integers in
            [0, max_seq_len), on a device supported for buffer indexing.
        """

        x_even = x[..., 0::2] # (B,L,D/2)
        x_odd = x[..., 1::2]

        cos = self.cos_table[token_positions].to(x.dtype) # (B,L,D/2) x.dtype
        sin = self.sin_table[token_positions].to(x.dtype) # (B,L,D/2) x.dtype

        y_even = cos*x_even  - sin*x_odd # (B,L,D/2) x.dtype x (B,L,D/2)
        y_odd = sin*x_even + cos*x_odd

        return torch.stack((y_even, y_odd), dim=-1).flatten(-2) # (B,L,D/2) --> (B,L,D/2,2) --> (B,L,D) x.dtype











    

