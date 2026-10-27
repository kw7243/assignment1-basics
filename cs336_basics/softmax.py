import math
from jaxtyping import Float, Int
from torch import Tensor

def softmax(x: Float[Tensor, " ... "], dim: int) -> Float[Tensor, " ... "]: 
    """
    Normalize the input along one dimension with a stable softmax.

    Args:
        x (Float[Tensor, "..."]): Unnormalized scores.
        dim (int): Dimension to normalize. Negative dimensions are supported.

    Returns:
        Float[Tensor, "..."]: Probabilities that preserve the input dtype and device.

    Requires:
        dim selects a nonempty dimension. Each slice has a finite maximum.
    """

    x = x - x.amax(dim=dim, keepdim=True) # (B,L,D) - (B,L,1) x.dtype

    exp = math.e**x  # (B,L,D) float32

    return exp / exp.sum(dim=dim, keepdim=True) # (B,L,D) / (B,L,1) float32
