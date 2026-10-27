import torch
import torch.nn as nn
import math
from jaxtyping import Float
from torch import Tensor

class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Create an RMSNorm layer.

        Args:
            d_model (int): Hidden dimension of the model.
            eps (float): Epsilon value for numerical stability.
            device (torch.device | str | None): Device for the gain parameter. None uses the PyTorch default.
            dtype (torch.dtype | None): Floating-point dtype for the gain. None uses the PyTorch default.
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

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        """
        Normalize the final dimension and apply the gain.

        Args:
            x (Float[Tensor, "... d_model"]): Must be on the gain's device.

        Returns:
            Float[Tensor, "... d_model"]: Preserves the input dtype and device.
                Normalization uses float32 intermediates, including for float64
                inputs. Gain multiplication follows PyTorch dtype promotion
                before conversion back to the input dtype.
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)# (B,L,1)
        x = self.gain * (x / rms) # (D,)x(B,L,D)

        return x.to(in_dtype)

    

