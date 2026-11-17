import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int
from cs336_basics.softmax import softmax
from cs336_basics.kv_cache import KVCache

def sample_next_token(
    logits: Float[Tensor, "batch_size vocab_size"],
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> Int[Tensor, "batch_size 1"]:
    """Sample one token independently for each batch element.

    Args:
        logits (Float[Tensor, "batch_size vocab_size"]): Next-token logits.
        temperature (float): Positive sampling temperature.
        top_p (float): Nucleus sampling threshold in (0, 1].

    Returns:
        Int[Tensor, "batch_size 1"]: One sampled token per sequence.
    """
    if temperature <= 0 or not 0 < top_p <= 1:
        raise ValueError("Invalid temperature or top_p")
    
    probs = softmax(logits / temperature, dim=-1) # (B,V)
    sorted_probs, sorted_ids = probs.sort(dim=-1, descending=True)

    # top_p
    keep_mask = sorted_probs.cumsum(dim=-1) - sorted_probs < top_p # (B,V)
    sorted_probs = sorted_probs * keep_mask

    # multinomial renormalizes already
    sampled_id = torch.multinomial(sorted_probs, num_samples=1) # (B,V) --> (B,1)
    next_token_id = torch.gather(input=sorted_ids, dim=-1, index=sampled_id)
    return next_token_id


@torch.no_grad()
def prefill(
    model: nn.Module,
    prompt_ids: Int[Tensor, "batch_size seq_len"],
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> tuple[Int[Tensor, "batch_size 1"], KVCache]:
    """Prefill the prompt and return the first sampled token and new cache."""
    batch_size, seq_len = prompt_ids.shape

    context_length = model.context_length
    if not isinstance(context_length, int):
        raise TypeError("model.context_length must be an integer.")
    if seq_len > context_length:
        raise ValueError("Prompt exceeds model context length")

    num_layers, num_heads, d_head = model.num_layers, model.num_heads, model.d_head
    if not isinstance(num_layers, int) or not isinstance(num_heads, int) or not isinstance(d_head, int):
        raise TypeError("Model layer and head dimensions must be integers.")

    reference = next((parameter for parameter in model.parameters() if parameter.is_floating_point()), None)
    if reference is None:
        reference = next((buffer for buffer in model.buffers() if buffer.is_floating_point()), None)
    if reference is None:
        raise ValueError("Model must have floating parameters or buffers for KV-cache storage.")

    kv_cache = KVCache(num_layers,
                      batch_size,
                      num_heads,
                      context_length,
                      d_head,
                      reference.device,
                      reference.dtype)
    
    # Forward pass + sample
    logits = model(token_ids=prompt_ids, 
                    kv_cache=kv_cache)[:, -1, :]  # (B,L,V) --> (B,V)
    
    next_token_id = sample_next_token(logits, temperature, top_p)  # (B,1) tower

    return next_token_id, kv_cache

@torch.no_grad()
def decode_step(
    model: nn.Module,
    token_ids: Int[Tensor, "batch_size 1"],
    kv_cache: KVCache,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> Int[Tensor, "batch_size 1"]:
    """Consume one token per sequence, update the cache, and sample the next."""
    if token_ids.ndim != 2 or token_ids.shape[-1] != 1:
        raise ValueError("decode_step expects one token per sequence.")

    # Forward pass + sample
    logits = model(token_ids=token_ids,
                   kv_cache=kv_cache)[:, -1, :]  # (B,1,V) --> (B,V)

    next_token_id = sample_next_token(logits, temperature, top_p)  # (B,1) tower

    return next_token_id

@torch.no_grad()
def generate_kv(
    model: nn.Module,
    prompt_ids: Int[Tensor, "batch_size seq_len"],
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
) -> Int[Tensor, "batch_size total_seq_len"]:
    """Generate tokens autoregressively with KV cache.

    Args:
        model (nn.Module): Language model with an integer context_length.
        prompt_ids (Int[Tensor, "batch seq_len"]): Batched token IDs.
        max_new_tokens (int): Maximum number of new tokens.
        temperature (float): Positive sampling temperature.
        top_p (float): Nucleus sampling threshold in (0, 1].
        eos_token_id (int | None): Optional end-of-sequence token ID.

    Returns:
        Int[Tensor, "batch total_seq_len"]: Prompt and generated
            tokens, with a shared output length across the batch.

    Requires:
        batch > 0 and seq_len > 0.
        max_new_tokens >= 0.
        All sequences have the same initial length.
        Prompt IDs and model are on compatible devices.

    Raises:
        TypeError: If model.context_length is not an integer.
    """
    context_length = model.context_length
    if not isinstance(context_length, int):
        raise TypeError("model.context_length must be an integer.")

    model.eval()
    
    batch_size, seq_len = prompt_ids.shape

    output_ids = torch.empty(
        (batch_size, seq_len + max_new_tokens),
        dtype=prompt_ids.dtype,
        device=prompt_ids.device,
    )  # (B, L + N)
    output_ids[:, :seq_len] = prompt_ids

    finished = torch.zeros(
        (batch_size, 1), dtype=torch.bool, device=prompt_ids.device
    )  # (B,1)

    length = seq_len

    kv_cache: KVCache | None = None

    for _ in range(max_new_tokens):
        if length > context_length:
            break

        # Forward pass + sample
        if kv_cache is None:
            # prefill
            next_token_id, kv_cache = prefill(model, prompt_ids, temperature, top_p)
        else:
            # decode
            next_token_id = decode_step(model, output_ids[:, length - 1:length], kv_cache, temperature, top_p)

        # Preserve EOS
        if eos_token_id is not None:
            next_token_id = torch.where(finished, eos_token_id, next_token_id) # (B,1)
            finished |= next_token_id == eos_token_id # (B,1)

        output_ids[:, length] = next_token_id.squeeze(-1) # (B,)
        length += 1

        # End early if all batch seqs finished
        if eos_token_id is not None and finished.all().item():
            break

    return output_ids[:, :length]




@torch.no_grad()
def generate_naive(
    model: nn.Module,
    prompt_ids: Int[Tensor, "batch_size seq_len"],
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
) -> Int[Tensor, "batch_size total_seq_len"]:
    """Generate tokens autoregressively without a KV cache.

    Args:
        model (nn.Module): Language model with an integer context_length.
        prompt_ids (Int[Tensor, "batch seq_len"]): Batched token IDs.
        max_new_tokens (int): Maximum number of new tokens.
        temperature (float): Positive sampling temperature.
        top_p (float): Nucleus sampling threshold in (0, 1].
        eos_token_id (int | None): Optional end-of-sequence token ID.

    Returns:
        Int[Tensor, "batch total_seq_len"]: Prompt and generated
            tokens, with a shared output length across the batch.

    Requires:
        batch > 0 and seq_len > 0.
        max_new_tokens >= 0.
        All sequences have the same initial length.
        Prompt IDs and model are on compatible devices.

    Raises:
        TypeError: If model.context_length is not an integer.
    """
    context_length = model.context_length
    if not isinstance(context_length, int):
        raise TypeError("model.context_length must be an integer.")

    model.eval()
    
    batch_size, seq_len = prompt_ids.shape

    output_ids = torch.empty(
        (batch_size, seq_len + max_new_tokens),
        dtype=prompt_ids.dtype,
        device=prompt_ids.device,
    )  # (B, L + N)
    output_ids[:, :seq_len] = prompt_ids

    finished = torch.zeros(
        (batch_size, 1), dtype=torch.bool, device=prompt_ids.device
    )  # (B,1)

    length = seq_len

    for _ in range(max_new_tokens):
        if length > context_length:
            break

        # Forward pass + sample
        logits = model(output_ids[:, :length])[:, -1, :]  # (B,L,V) --> (B,V)
        next_token_id = sample_next_token(logits, temperature, top_p)  # (B,1) tower

        # Preserve EOS
        if eos_token_id is not None:
            next_token_id = torch.where(finished, eos_token_id, next_token_id) # (B,1)
            finished |= next_token_id == eos_token_id # (B,1)

        output_ids[:, length] = next_token_id.squeeze(-1) # (B,)
        length += 1

        # End early if all batch seqs finished
        if eos_token_id is not None and finished.all().item():
            break

    return output_ids[:, :length]


