
import torch
import torch.nn as nn
import math
from jaxtyping import Float, Int, Bool
from torch import Tensor
from cs336_basics.embedding import Embedding
from cs336_basics.rope import RotaryPositionalEmbedding
from cs336_basics.transformer import TransformerBlock
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.linear import Linear
from cs336_basics.softmax import softmax

class TransformerLM(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int, 
        rope_theta: float,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Create 

        Args:
            d_model (int): Positive model dimension, divisible by num_heads.
            num_heads (int): Positive number of attention heads.
            d_ff (int): Intermediate dimension.
            rope_theta (float): 
            vocab_size: int
            context_length: int,
            num_layers: int,
            device (torch.device | str | None): Device for the projection parameters.
            dtype (torch.dtype | None): Floating-point dtype for the projection parameters.

        Requires:
            pos_layer, when supplied, is compatible with the projection device
            and dtype. These arguments do not move or cast pos_layer.

        Raises:
            ValueError: If either dimension is not positive, or d_model is not
                divisible by num_heads.
        """
        super().__init__()

        if d_model <= 0 or num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError("d_model and num_heads must be positive, and d_model must be divisible by num_heads.")

        # Module params
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_head = self.d_model // num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers

        self.rope = RotaryPositionalEmbedding(rope_theta, self.d_head, 
                                        context_length, 
                                        device)

        # Transformer model
        self.token_embeddings = Embedding(num_embeddings=vocab_size, 
                                          embedding_dim=d_model,
                                          device=device,
                                          dtype=dtype)

        self.layers = nn.ModuleList([TransformerBlock(self.d_model, 
                                                      self.num_heads, 
                                                      self.d_ff, 
                                                      self.rope,
                                                      device,
                                                      dtype,)
                                    for _ in range(self.num_layers)])
        
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)


    def forward(
        self,
        token_ids: Int[Tensor, "... seq_len"],
        token_positions: Int[Tensor, "... seq_len"] | None = None,
        mask: Bool[Tensor, "..."] | None = None,
    ) -> Float[Tensor, "... seq_len vocab_size"]:
        """

        Args:
            token_ids (Int[Tensor, "... seq_len d_model"]): Input representations. seq_len <= `self.context_length`
            token_positions (Int[Tensor, "... seq_len"] | None): Required when
                pos_layer is supplied; otherwise unused.
            mask (Bool[Tensor, "..."] | None): Broadcasts to the attention scores
                (..., num_heads, seq_len, seq_len). True allows a key position.
                The causal mask also applies.

        Returns:
            Float[Tensor, "... seq_len vocab_size"]: Unnormalized logits for distribution of next token.

        Requires:
            token_ids is compatible with the projection parameter dtype and device.
            mask, when supplied, is on the same device as x.
            token_positions meets the positional layer's shape, index, and
            device requirements. Each query has at least one allowed key.

        Raises:
            ValueError: If the input width does not match d_model, or positions
                are missing when a positional layer is supplied.
        """
        x = self.token_embeddings(token_ids)

        for layer in self.layers:
            x = layer(x, token_positions, mask)

        x = self.ln_final(x)    

        return self.lm_head(x)
        
