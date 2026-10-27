
import math

import torch
from torch import nn


class Linear(nn.Module):
    """A bias-free linear transformation."""

    # Rep:
    #   in_features: Input dimension.
    #   out_features: Output dimension.
    #   weight: Trainable tensor of shape
    #       (out_features, in_features).
    #
    # Rep invariant:
    #   in_features > 0
    #   out_features > 0
    #   weight is a registered nn.Parameter
    #   weight.requires_grad is True
    #   weight.shape == (out_features, in_features)
    #
    # Abstraction function:
    #   AF(weight) = the linear map f(X) = X @ weight.T.
    #
    # Safety from rep exposure:
    #   PyTorch intentionally exposes mutable parameters.
    #   External code can modify weight, so the class
    #   cannot independently guarantee its rep invariant.
    #   Clients must preserve its shape and registration.

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Construct a bias-free linear layer.

        Args:
            in_features (int): Positive input dimension.
            out_features (int): Positive output dimension.
            device (torch.device | str | None): Device for the weight parameter. None uses the PyTorch default.
            dtype (torch.dtype | None): Floating-point dtype for the weights. None uses the PyTorch default.

        Raises:
            ValueError: If a feature dimension is not positive.
        """
        super().__init__()

        if in_features <= 0 or out_features <= 0:
            raise ValueError("Feature dimensions must be positive.")

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty(
                (out_features, in_features),
                device=device,
                dtype=dtype,
            )
        )

        std = math.sqrt(2.0 / (in_features + out_features))

        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear transformation.

        Args:
            x (Float[Tensor, "... in_features"]): Activations to transform.

        Returns:
            Float[Tensor, "... out_features"]: Uses the input device. With matching
                input and weight dtypes, preserves that dtype outside autocast.

        Requires:
            x is compatible with the weight dtype and device.
        """
        return x @ self.weight.T
