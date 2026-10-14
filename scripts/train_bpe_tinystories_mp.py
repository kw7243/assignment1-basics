import argparse
import os
import pickle
import time
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

from cs336_basics.bpe import train_bpe, count_pretokens


SPECIAL_TOKENS = ["<|endoftext|>"]


def find_chunk_boundaries(
    file,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Find chunk boundaries aligned with a special token.

    Adapted from Stanford CS336 pretokenization_example.py.
    """
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    boundaries = [
        i * chunk_size
        for i in range(desired_num_chunks + 1)
    ]
    boundaries[-1] = file_size

    for bi in range(1, len(boundaries) - 1):
        position = boundaries[bi]
        file.seek(position)

        while True:
            data = file.read(4096)

            if not data:
                boundaries[bi] = file_size
                break

            index = data.find(split_special_token)

            if index != -1:
                boundaries[bi] = position + index
                break

            position += 4096

    return sorted(set(boundaries))


def count_chunk(args):
    """
    Worker: read one chunk and count its pre-tokens.
    """
    input_path, start, end = args

    with open(input_path, "rb") as f:
        f.seek(start)
        corpus = f.read(end - start).decode("utf-8")

    return count_pretokens(corpus, SPECIAL_TOKENS)


def main():
    # Parse arguments
    input_path = Path("data/TinyStoriesV2-GPT4-train.txt")
    output_prefix = Path("data/tinystories_bpe_mp")
    vocab_path = output_prefix.with_name(f"{output_prefix.name}_vocab.pkl")
    merges_path = output_prefix.with_name(f"{output_prefix.name}_merges.pkl")

    parser = argparse.ArgumentParser(
        description="Train TinyStories BPE with multiprocessing."
    )

    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Number of worker processes (default: min(4, CPU count))",
    )

    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1")

    num_workers = args.workers

    print("Starting BPE training on TinyStories...")
    start_time = time.perf_counter()

    # Step 1: Find document-aligned chunk boundaries.
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f,
            desired_num_chunks=num_workers,
            split_special_token=b"<|endoftext|>",
        )

    tasks = [
        (input_path, start, end)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    # Step 2: Parallel pre-tokenization.
    corpus_freq = Counter()

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        for partial_freq in pool.map(count_chunk, tasks):
            corpus_freq.update(partial_freq)

    pretoken_time = time.perf_counter() - start_time

    # Step 3: Train BPE using your existing merge algorithm.
    merge_start = time.perf_counter()

    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=10_000,
        special_tokens=SPECIAL_TOKENS,
        corpus_freq=corpus_freq,
    )

    merge_time = time.perf_counter() - merge_start

    # Step 4: Serialize results.
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    with open(merges_path, "wb") as f:
        pickle.dump(merges, f)

    print(f"Workers: {num_workers}")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")
    print(f"Pre-tokenization: {pretoken_time:.2f}s")
    print(f"BPE merging: {merge_time:.2f}s")
    print(f"Total: {time.perf_counter() - start_time:.2f}s")
    print(f"Vocabulary saved to: {vocab_path}")
    print(f"Merges saved to: {merges_path}")


if __name__ == "__main__":
    main()
