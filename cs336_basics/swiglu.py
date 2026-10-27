import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor
from cs336_basics.linear import Linear

class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Create a SwiGLU feed-forward layer.

        Args:
            d_model (int): Input and output dimension of the model.
            d_ff (int): Intermediate dimension.
            device (torch.device | str | None): Device for all projection parameters. None uses the PyTorch default.
            dtype (torch.dtype | None): Floating-point dtype for all projections. None uses the PyTorch default.
        """
        super().__init__()

        # Module params
        self.up_proj = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.gate_proj = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.down_proj = Linear(d_ff, d_model, device=device, dtype=dtype)


    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        """
        Apply the gated feed-forward transformation.

        Args:
            x (Float[Tensor, "... d_model"]): Must have a dtype and device
                compatible with all projection parameters.

        Returns:
            Float[Tensor, "... d_model"]: Uses the input device. With matching
                input and projection dtypes, preserves that dtype outside autocast.
        """
        gate = self.gate_proj(x) # (B,L,D)x(D,I)
        silu = gate*torch.sigmoid(gate) # (B,L,I)

        up = self.up_proj(x) # (B,L,I)

        return self.down_proj(silu * up) # (B,L,I)x(I,D)

    

