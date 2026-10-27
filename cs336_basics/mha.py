
import torch
import torch.nn as nn
import math
from jaxtyping import Float, Int, Bool
from torch import Tensor
from cs336_basics.linear import Linear
from cs336_basics.sdpa import sdpa
from cs336_basics.rope import RotaryPositionalEmbedding

class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, pos_layer: nn.Module | None = None, device=None, dtype=None) -> None:
        """
        Create MHA layer.

        Args:
            d_model: int  Hidden dimension of the model
            num_heads (int)
            positional_layer (nn.Module | None): positional layer to apply to q,k (ex. RoPE)
            device: torch.device | None = None  Device to store the parameters on
            dtype: torch.dtype | None = None  Data type of the parameters
        """
        super().__init__()

        assert d_model > 0 and d_model % num_heads == 0, "d_model not divisible by hedas"

        # Module params
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_head = self.d_model // num_heads

        self.q_proj = Linear(self.d_model, self.d_model, device, dtype)
        self.k_proj = Linear(self.d_model, self.d_model, device, dtype)
        self.v_proj = Linear(self.d_model, self.d_model, device, dtype)
        self.o_proj = Linear(self.d_model, self.d_model, device, dtype)

        self.pos_layer = pos_layer


    def forward(self, x: Float[Tensor, " ... seq_len d_model"], 
                token_positions: Int[Tensor, " ... seq_len"] | None = None,
                mask: Bool [Tensor, "seq_len seq_len"] | None = None) -> Float[Tensor, " ... seq_len d_k"]:
        """
        Args:
            x (Float[Tensor, "... seq_len d_k"]): input tensor (ex. (B,L,D))
            token_positions (Int[Tensor, " ... seq_len"]): token positions for a batch 
                                            (diff reqs process seqs at different positions)
        Returns:
            Float[Tensor, "... seq_len d_k"]: MHA(x)
                                            ex. (B,L,D)
        """
        *batch_dims, seq_len, d_model = x.shape
        assert d_model == self.d_model, "Input tensor dims don't match"

        # Q,K,V proj
        q = self.q_proj(x) # (B,L,D)x(D,D)
        k = self.k_proj(x) 
        v = self.v_proj(x)

        q = q.reshape(*batch_dims, seq_len, self.num_heads, self.d_head).transpose(-2,-3) # (B,H,L,d)
        k = k.reshape(*batch_dims, seq_len, self.num_heads, self.d_head).transpose(-2,-3) # (B,H,L,d)
        v = v.reshape(*batch_dims, seq_len, self.num_heads, self.d_head).transpose(-2,-3) # (B,H,L,d)

        # positional encodings
        if self.pos_layer:
            assert token_positions is not None, "Missing token_positions"
            tok_pos = token_positions.unsqueeze(-2) # (B,L) --> (B,1,L)
            q, k = self.pos_layer(q, tok_pos), self.pos_layer(k, tok_pos)

        # masking
        causal_mask = torch.ones(
            seq_len, seq_len,
            dtype=torch.bool,
            device=x.device,
        ).tril()

        if mask is not None:
            causal_mask = causal_mask & mask

        # sdpa
        mh_sdpa = sdpa(q, k, v, causal_mask) # (B,H,L,d)x(B,H,d,L)x(B,H,L,d) = (B,H,L,d)
        mh_sdpa = mh_sdpa.transpose(-2,-3).reshape(*batch_dims, seq_len, self.d_model) # (B,L,H*d)=(B,L,D)

        return self.o_proj(mh_sdpa) # (B,L,D)x(D,D)
        










    

