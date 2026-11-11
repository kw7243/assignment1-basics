
import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int
from cs336_basics.softmax import softmax

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
        model (nn.Module): Transformer language model.
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
    """
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
        if length > model.context_length:
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
