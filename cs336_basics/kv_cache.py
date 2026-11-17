import torch

from jaxtyping import Bool, Float, Int
from torch import Tensor


class KVCache:
    """KV-cache storage for autoregressive Transformer inference."""

    # Rep:
    #   keys, values: Lists of tensors, one per layer.
    #       Each tensor has shape (B, H_kv, max_seq_len, d_head).
    #   lengths: Number of valid cached tokens per layer.
    #
    # Rep invariant:
    #   len(keys) == len(values) == len(lengths) == num_layers
    #   keys[i].shape == values[i].shape
    #   0 <= lengths[i] <= max_seq_len
    #   All tensors have the configured device and dtype.
    #
    # Abstraction function:
    #   For layer i, keys[i][:, :, :lengths[i], :] and
    #   values[i][:, :, :lengths[i], :] represent the
    #   cached K/V vectors for its processed token prefix.
    #
    # Safety from rep exposure:
    #   get() returns tensor views into internal storage.
    #   Clients must not modify the returned tensors.

    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        num_kv_heads: int,
        max_seq_len: int,
        d_head: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """Allocate KV-cache storage.

        Args:
            num_layers: Number of Transformer layers.
            batch_size: Number of sequences in the batch.
            num_kv_heads: Number of key/value heads per layer.
            max_seq_len: Maximum cached sequence length.
            d_head: Dimension of each key/value head.
            device: Storage device.
            dtype: Storage data type.
        """
        shape = (batch_size, num_kv_heads, max_seq_len, d_head)

        self.kv = [
            (
                torch.empty(shape, device=device, dtype=dtype),
                torch.empty(shape, device=device, dtype=dtype),
            )
            for _ in range(num_layers)
]

        self.lengths = [0]*num_layers

        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_kv_heads = num_kv_heads
        self.max_seq_len = max_seq_len
        self.d_head = d_head


    def get(self, layer_idx: int) -> tuple[Float[Tensor, "batch_size num_heads seq_len d_head"], 
                                           Float[Tensor, "batch_size num_heads seq_len d_head"]]:
        """Return the valid cached K/V prefix for one layer.

        Args:
            layer_idx: Index of the Transformer layer.

        Returns:
            K, V: Views of shape (B, H_kv, L, d_head),
                where L is the layer's current cache length.
        """
        if layer_idx >= self.num_layers:
            raise ValueError

        k, v = self.kv[layer_idx]
        seq_len = self.lengths[layer_idx]

        return k[..., :seq_len, :], v[..., :seq_len, :]

    def append(
        self,
        layer_idx: int,
        k_new: Float[Tensor, "batch_size num_heads T d_head"],
        v_new: Float[Tensor, "batch_size num_heads T d_head"],
    ) -> None:
        """Write new keys and values into preallocated storage.

        Args:
            layer_idx: Index of the Transformer layer.
            k_new: New keys of shape (batch_size, num_heads, T, d_head).
            v_new: New values of shape (batch_size, num_heads, T, d_head).

        Requires:
            lengths[layer_idx] + T <= max_seq_len.

        Effects:
            Writes K/V into positions [start_pos, start_pos + T).
            Increases the valid length of the given layer.
        """
        if not 0 <= layer_idx < self.num_layers:
            raise IndexError("Invalid layer index.")

        if k_new.shape != v_new.shape:
            raise ValueError("New keys and values' shape don't match each other")
        

        B, H, T, D = k_new.shape

        if (B, H, D) != (
            self.batch_size,
            self.num_kv_heads,
            self.d_head,
        ):
            raise ValueError("Invalid K/V dimensions.")


        seq_len = self.lengths[layer_idx]
        _, _, new_len, _ = k_new.shape

        if seq_len + new_len > self.max_seq_len:
            raise ValueError("New kv addition exceeds max_seq_len")

        # update kv
        k, v = self.kv[layer_idx]
        k[..., seq_len: seq_len + new_len, :] = k_new
        v[..., seq_len: seq_len + new_len, :] = v_new

        # update lengths
        self.lengths[layer_idx] += new_len

        return None

        
    def get_length(self, layer_idx: int) -> int:
        """Return the number of valid cached tokens in a layer."""
        if not 0 <= layer_idx < self.num_layers:
            raise IndexError("Invalid layer index.")

        return self.lengths[layer_idx]

    def clear(self) -> None:
        """Reset all cache lengths without reallocating storage."""
        self.lengths = [0]*self.num_layers
