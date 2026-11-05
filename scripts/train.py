"""Train a CS336-style decoder-only Transformer on pretokenized data.

Independent-study reference for CS336 A1, Section 5.3.

Integration points:
    - cs336_basics.transformer_lm.TransformerLM
    - cs336_basics.optimizers.AdamW
    - cs336_basics.data.get_batch
    - cs336_basics.loss.cross_entropy

Data formats:
    .npy: NumPy array saved using np.save (memory-mapped by np.load).
    .bin: Raw token IDs (memory-mapped with --raw-dtype, default uint16).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import wandb
from torch import Tensor, nn

# Use the existing model, loss, and training utilities.
from cs336_basics.data import get_batch
from cs336_basics.loss import cross_entropy
from cs336_basics.optimizers import AdamW
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.schedulers import get_lr_cosine_schedule
from cs336_basics.optimizers import clip_gradients
from cs336_basics.checkpointing import save_checkpoint, load_checkpoint

@dataclass(frozen=True)
class TrainConfig:
    """Configuration for a single-model, single-device training run.

    Invariant:
        Model dimensions and training counts are positive; d_model is
        divisible by num_heads; warmup_steps < num_steps; learning rates
        satisfy 0 <= min_lr <= max_lr.
    """

    train_data: Path
    val_data: Path
    checkpoint_dir: Path
    resume: Path | None = None
    raw_dtype: str = "uint16"
    device: str = "cpu"
    seed: int = 42

    vocab_size: int = 10_000
    context_length: int = 256
    d_model: int = 512
    num_heads: int = 16
    d_ff: int = 1344
    num_layers: int = 4
    rope_theta: float = 10_000.0

    batch_size: int = 32
    num_steps: int = 2_000
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 100
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0

    log_every: int = 20
    val_every: int = 100
    val_batches: int = 20
    save_every: int = 500

    def validate(self) -> None:
        """Validate configuration values before allocating model state."""
        positive = {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "d_ff": self.d_ff,
            "num_layers": self.num_layers,
            "batch_size": self.batch_size,
            "num_steps": self.num_steps,
            "log_every": self.log_every,
            "val_every": self.val_every,
            "val_batches": self.val_batches,
            "save_every": self.save_every,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive: {value}")
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if not 0 <= self.warmup_steps < self.num_steps:
            raise ValueError("Require 0 <= warmup_steps < num_steps")
        if not 0 <= self.min_lr <= self.max_lr:
            raise ValueError("Require 0 <= min_lr <= max_lr")
        if not (0 <= self.beta1 < 1 and 0 <= self.beta2 < 1):
            raise ValueError("AdamW betas must lie in [0, 1)")
        if self.eps <= 0 or self.weight_decay < 0 or self.max_grad_norm <= 0:
            raise ValueError("eps and max_grad_norm must be positive; weight_decay >= 0")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")
        if np.dtype(self.raw_dtype).kind not in "iu":
            raise ValueError("raw_dtype must be an integer dtype")
        for path in (self.train_data, self.val_data):
            if not path.is_file():
                raise FileNotFoundError(path)
        if self.resume is not None and not self.resume.is_file():
            raise FileNotFoundError(self.resume)


def load_token_ids(path: Path, raw_dtype: str, context_length: int, vocab_size: int) -> np.ndarray:
    """Open a disk-backed one-dimensional token array without loading it all.

    Args:
        path: .npy file from np.save, or headerless .bin token file.
        raw_dtype: On-disk dtype used only for .bin files.
        context_length: Required number of input positions per example.
        vocab_size: Used for inexpensive sampled range validation.

    Returns:
        Read-only NumPy memory map with integer tokens.

    Raises:
        ValueError: If format, dimensions, or sampled token IDs are invalid.
    """
    if path.suffix == ".npy":
        data = np.load(path, mmap_mode="r", allow_pickle=False)
    elif path.suffix == ".bin":
        data = np.memmap(path, dtype=np.dtype(raw_dtype), mode="r")
    else:
        raise ValueError(f"Expected .npy or .bin, got: {path}")

    if data.ndim != 1 or data.dtype.kind not in "iu":
        raise ValueError(f"Expected 1D integer token IDs, got {data.shape} {data.dtype}")
    if len(data) <= context_length:
        raise ValueError(f"Dataset must contain at least {context_length + 1} tokens")

    # Only probes a small subset. This does not validate every token on disk.
    locations = np.linspace(0, len(data) - 1, num=min(64, len(data)), dtype=np.int64)
    observed = data[locations]
    if np.any(observed >= vocab_size) or np.any(observed < 0):
        raise ValueError(f"Sampled token IDs fall outside [0, {vocab_size})")
    return data


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    data: np.ndarray,
    config: TrainConfig,
    device: torch.device,
) -> float:
    """Estimate mean per-token validation loss on sampled windows.

    Restores the model's previous train/eval mode even if evaluation fails.
    """
    # Restore the previous mode after validation.
    was_training = model.training
    model.eval()
    total = torch.zeros((), device=device)
    try:
        for _ in range(config.val_batches):
            inputs, labels = get_batch(data, config.batch_size, config.context_length, str(device))
            logits = model(inputs)
            total += cross_entropy(logits, labels).detach()
    finally:
        model.train(was_training)
    return (total / config.val_batches).item()


def _rng_state() -> dict[str, Any]:
    """Capture random generator states in a weights_only-loadable format."""
    np_state = np.random.get_state()
    # This serialization format requires the legacy tuple state.
    assert isinstance(np_state, tuple), "Expected a legacy NumPy RNG state tuple"
    np_name, np_keys, np_position, np_has_gauss, np_gaussian = np_state
    return {
        "numpy": (np_name, np_keys.tolist(), int(np_position), int(np_has_gauss), float(np_gaussian)),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(saved: dict[str, Any]) -> None:
    """Restore RNG states for reproducible single-process resumption."""
    np_name, np_keys, np_position, np_has_gauss, np_gaussian = saved["numpy"]
    np.random.set_state(
        (np_name, np.asarray(np_keys, dtype=np.uint32), np_position, np_has_gauss, np_gaussian)
    )
    torch.set_rng_state(saved["torch"].cpu())
    if torch.cuda.is_available() and len(saved["cuda"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(saved["cuda"])


def _serialized_config(config: TrainConfig) -> dict[str, Any]:
    """Return a checkpoint-safe configuration dictionary."""
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }



def write_metric(path: Path, record: dict[str, Any]) -> None:
    """Append a JSON metrics record for later plotting and comparison."""
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, sort_keys=True) + "\n")


def train(config: TrainConfig) -> None:
    """Run configurable single-device autoregressive language-model training."""
    config.validate()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Load datasets as np_arr token_ids (ex. [0,43,52,792,...])
    train_data = load_token_ids(config.train_data, config.raw_dtype, config.context_length, config.vocab_size)
    val_data = load_token_ids(config.val_data, config.raw_dtype, config.context_length, config.vocab_size)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create model, optimizer from config
    device = torch.device(config.device)
    model = TransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=config.d_model,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        num_layers=config.num_layers,
        rope_theta=config.rope_theta,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=config.max_lr,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay,
    )

    # Prep training loop
    start_t = 0
    if config.resume is not None:
        # The checkpoint stores the index of the next step to execute.
        start_t = load_checkpoint(config.resume, model, optimizer)
        print(f"Resuming at step {start_t}", flush=True)
    model.train()

    with wandb.init(project="cs336", config=_serialized_config(config)) as wandb_run:
        wandb_run.define_metric("t")
        wandb_run.define_metric("*", step_metric="t")

        metrics_path = config.checkpoint_dir / "metrics.jsonl"
        window_loss = torch.zeros((), device=device)
        window_steps = 0
        window_start = time.perf_counter()

        ## TRAINING LOOP ##
        # t is the zero-based step index; num_steps is the total update count.
        for t in range(start_t, config.num_steps):
            # Get lr for update
            lr = get_lr_cosine_schedule(
                t,
                max_learning_rate=config.max_lr,
                min_learning_rate=config.min_lr,
                warmup_iters=config.warmup_steps,
                cosine_cycle_iters=config.num_steps,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr

            ## MAIN LOOP ##
            inputs, targets = get_batch(train_data, 
                                        config.batch_size, 
                                        config.context_length, 
                                        config.device) # (B,L)

            # forward pass to create comp graph
            optimizer.zero_grad()
            logits = model(inputs) # (B,L,V)
            loss = cross_entropy(logits, targets) # ()

            # populate grads in model.state_dict()
            loss.backward() 
            clip_gradients(model.parameters(), config.max_grad_norm) # optional grad clip

            # optimizer.param_groups and .state to update
            optimizer.step() 


            ## LOGGING AND VAL ##
            # Accumulate loss without retaining the computation graph.
            window_loss += loss.detach()
            window_steps += 1
            
            # periodic logging
            should_log = t % config.log_every == 0 or t == config.num_steps - 1
            if should_log:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                wall_seconds = max(time.perf_counter() - window_start, 1e-9)
                record = {
                    "t": t,
                    "train/loss": (window_loss / window_steps).item(),
                    "train/lr": lr,
                    "train/tokens_per_second": (window_steps * config.batch_size * config.context_length) / wall_seconds,
                }
                if not math.isfinite(record["train/loss"]):
                    raise FloatingPointError(f"Nonfinite training loss at step {t}")
                print(json.dumps(record, sort_keys=True), flush=True)
                write_metric(metrics_path, record)
                wandb_run.log(record)
                window_loss.zero_()
                window_steps = 0

            # periodic validation
            should_val = t % config.val_every == 0 or t == config.num_steps - 1
            if should_val:
                val_loss = evaluate(model, val_data, config, device)
                record = {"t": t, "val/loss": val_loss, "val/perplexity": math.exp(val_loss)}
                print(json.dumps(record, sort_keys=True), flush=True)
                write_metric(metrics_path, record)
                wandb_run.log(record)

            # periodic checkpointing
            if t % config.save_every == 0 or t == config.num_steps - 1:
                # Save the model, optimizer, and index of the next step to execute.
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    iteration=t + 1,
                    out=config.checkpoint_dir / "latest.pt",
                )

            if should_log: # restart window_start
                window_start = time.perf_counter()

def parse_args() -> TrainConfig:
    """Parse command-line options into an immutable, validated configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--val-data", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--raw-dtype", default="uint16", help="dtype for headerless .bin files")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, cuda:0, or mps")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-steps", "--max-steps", dest="num_steps", type=int, default=2_000,
                        help="total number of optimizer steps")
    parser.add_argument("--max-lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--val-every", type=int, default=100)
    parser.add_argument("--val-batches", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=500)
    return TrainConfig(**vars(parser.parse_args()))


def main() -> None:
    """Command-line entry point."""
    train(parse_args())


if __name__ == "__main__":
    main()
