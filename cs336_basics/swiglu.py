import torch
import torch.nn as nn
import math
from jaxtyping import Float, Int
from torch import Tensor
from cs336_basics.linear import Linear

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None) -> None:
        """
        Create RMSNorm layer.

        Args:
            d_model: int  Hidden dimension of the model
            d_ff (int): intermediate dim
            device: torch.device | None = None  Device to store the parameters on
            dtype: torch.dtype | None = None  Data type of the parameters
        """
        super().__init__()

        # Module params
        self.up_proj = Linear(d_model, d_ff) 
        self.gate_proj = Linear(d_model, d_ff)
        self.down_proj = Linear(d_ff, d_model)


    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        """
        Args:
            x (Int[Tensor, "... d_model"]): input tensor (ex. (B,L,D))

        Returns:
            Float[Tensor, "... d_model"]: SwiGLU(x) = (xW_up * SiLU(xW_gate))@W_down
                                            ex. (B,L,D)
        """
        gate = self.gate_proj(x) # (B,L,D)x(D,I)
        silu = gate*torch.sigmoid(gate) # (B,L,I)

        up = self.up_proj(x) # (B,L,I)

        return self.down_proj(silu * up) # (B,L,I)x(I,D)

    

