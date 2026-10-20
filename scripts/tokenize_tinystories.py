
import pickle
import time
from pathlib import Path

import numpy as np

from cs336_basics.tokenizer import Tokenizer


def tokenize_file(
    tokenizer: Tokenizer,
    input_path: Path,
    output_path: Path,
    chunk_size: int = 100_000,
) -> None:
    """
    Tokenize a text file and save token IDs as uint16.

    The output is a raw binary file readable by np.memmap.
    """
    start = time.perf_counter()
    token_ids = []
    total_tokens = 0

    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "wb") as f_out:

        for token_id in tokenizer.encode_iterable(f_in):
            token_ids.append(token_id)

            if len(token_ids) >= chunk_size:
                np.asarray(
                    token_ids, dtype=np.uint16
                ).tofile(f_out)

                total_tokens += len(token_ids)
                token_ids.clear()

        # Write remaining tokens.
        if token_ids:
            np.asarray(
                token_ids, dtype=np.uint16
            ).tofile(f_out)

            total_tokens += len(token_ids)

    elapsed = time.perf_counter() - start

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Tokens: {total_tokens:,}")
    print(f"Time: {elapsed:.2f}s")
    print()


def main():
    data_dir = Path("data")

    # Load your trained TinyStories tokenizer.
    with open(data_dir / "tinystories_bpe_mp_vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
        
    with open(data_dir / "tinystories_bpe_mp_merges.pkl", "rb") as f:
        merges = pickle.load(f)


    tokenizer = Tokenizer(
        vocab=vocab,
        merges=merges,
        special_tokens=["<|endoftext|>"],
    )

    # uint16 can represent token IDs from 0 to 65535.
    assert max(tokenizer.vocab) < 65536

    # Tokenize training dataset.
    tokenize_file(
        tokenizer,
        data_dir / "TinyStoriesV2-GPT4-train.txt",
        data_dir / "tinystories_train.bin",
    )

    # Tokenize validation dataset.
    tokenize_file(
        tokenizer,
        data_dir / "TinyStoriesV2-GPT4-valid.txt",
        data_dir / "tinystories_valid.bin",
    )


if __name__ == "__main__":
    main()
