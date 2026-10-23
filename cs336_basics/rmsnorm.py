import torch
import torch.nn as nn
import math
from jaxtyping import Float, Int
from torch import Tensor

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None) -> None:
        """
        Create RMSNorm layer.

        Args:
            d_model: int  Hidden dimension of the model
            eps: float = 1e-5  Epsilon value for numerical stability
            device: torch.device | None = None  Device to store the parameters on
            dtype: torch.dtype | None = None  Data type of the parameters
        """
        super().__init__()

        # Module params
        self.gain = nn.Parameter(torch.empty((d_model), 
                                device=device, 
                                dtype=dtype))
        self.eps = eps

        std = math.sqrt(2.0 / (d_model))
        nn.init.trunc_normal_(
            self.gain,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )

    def forward(self, x: Int[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        """
        Args:
            x (Int[Tensor, "... d_model"]): input tensor (B,L,D)

        Returns:
            Float[Tensor, "... d_model"]: RMSNorm(x) (B,L,D)
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)# (B,L,1)
        x = self.gain * (x / rms) # (D,)x(B,L,D)

        return x.to(in_dtype)

    

