import torch
import torch.nn as nn
import math
from jaxtyping import Float, Int32, Int64
from torch import Tensor

class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Create an embedding layer.

        Args:
            num_embeddings (int): Vocabulary size.
            embedding_dim (int): Model dimension, d_model.
            device (torch.device | str | None): Device for the embedding parameter. None uses the PyTorch default.
            dtype (torch.dtype | None): Floating-point dtype for the embedding parameter. None uses the PyTorch default.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), 
                                                   device=device, 
                                                   dtype=dtype))
        std = math.sqrt(2.0 / (num_embeddings + embedding_dim))
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )

    def forward(
        self, token_ids: Int32[Tensor, "..."] | Int64[Tensor, "..."]
    ) -> Float[Tensor, "... embedding_dim"]:
        """
        Get the embedding vector for each token ID.

        Args:
            token_ids (Int32[Tensor, "..."] | Int64[Tensor, "..."]): Values must be
                in [0, num_embeddings). Indices must be on the CPU
                or on the embedding parameter's device.

        Returns:
            Float[Tensor, "... embedding_dim"]: Uses the embedding parameter's
                dtype and device.
        """
        return self.weight[token_ids]

    


