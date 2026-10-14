
import pickle
import time
from pathlib import Path

from cs336_basics.bpe import train_bpe

def main():
    input_path = Path("data/TinyStoriesV2-GPT4-train.txt")
    output_prefix = Path("data/tinystories_bpe_naive")
    vocab_path = output_prefix.with_name(f"{output_prefix.name}_vocab.pkl")
    merges_path = output_prefix.with_name(f"{output_prefix.name}_merges.pkl")

    print("Starting BPE training...")
    
    start = time.perf_counter()

    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=10_000,
        special_tokens=["<|endoftext|>"],
    )

    elapsed = time.perf_counter() - start

    # Save the vocabulary and merges.
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    with open(merges_path, "wb") as f:
        pickle.dump(merges, f)

    print(f"Vocabulary size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")
    print(f"Training time: {elapsed:.2f}s")
    print(f"Vocabulary saved to: {vocab_path}")
    print(f"Merges saved to: {merges_path}")


if __name__ == "__main__":
    main()
