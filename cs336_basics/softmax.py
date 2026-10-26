import math
from jaxtyping import Float, Int
from torch import Tensor

def softmax(x: Float[Tensor, " ... "], dim: int) -> Float[Tensor, " ... "]: 
    """
    Args:
        x (Float[Tensor, "... "]): input tensor (ex. (B,L,D))
        dim: dimension to apply softmax over
    Returns:
        Float[Tensor, "... "] of same shape
    """

    x = x - x.amax(dim=dim, keepdim=True) # (B,L,D) - (B,L,1) x.dtype

    exp = math.e**x  # (B,L,D) float32

    return exp / exp.sum(dim=dim, keepdim=True) # (B,L,D) / (B,L,1) float32
