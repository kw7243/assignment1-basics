import torch
import torch.nn as nn
import math
from jaxtyping import Float, Int
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
            num_embeddings (int): The size of the vocabulary.
            embedding_dim (int): The size of each embedding vector, called d_model.
            device (torch.device | str | None): The device for the parameters.
            dtype (torch.dtype | None): The data type for the parameters.
        """
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((num_embeddings, embedding_dim), 
                                                   device=device, 
                                                   dtype=dtype))
        std = math.sqrt(2.0 / (num_embeddings + embedding_dim))
        nn.init.trunc_normal_(
            self.embedding,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )

    def forward(self, token_ids: Int[Tensor, " ..."]) -> Float[Tensor, " ... embedding_dim"]:
        """
        Get the embedding vector for each token ID.

        Args:
            token_ids (Int[Tensor, "..."]): Token IDs with any number of dimensions.

        Returns:
            Float[Tensor, "... embedding_dim"]: Embeddings with the input dimensions
                followed by embedding_dim
        """
        return self.embedding[token_ids]

    


