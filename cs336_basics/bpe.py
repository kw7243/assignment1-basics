import os
import regex as re

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a byte-level BPE tokenizer.

    Args:
        input_path: Path to training text.
        vocab_size: Maximum vocabulary size, including special tokens.
        special_tokens: Tokens that must never be split.

    Returns:
        vocab:
            Maps token ID -> bytes.
        merges:
            Ordered list of learned byte-pair merges.
    """

    # Step 1: Initialize vocabulary.
    vocab: dict[int, bytes] = {
        i: bytes([i]) for i in range(256)
    }

    for token in dict.fromkeys(special_tokens):
        vocab[len(vocab)] = token.encode("utf-8")


    # Step 2: Pre-tokenize the input corpus.
    with open(input_path, "r", encoding="utf-8") as f:
        corpus = f.read()

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    # Split at special-token boundaries.
    if special_tokens:
        escaped = [
            re.escape(token)
            for token in sorted(set(special_tokens), key=len, reverse=True)
        ]
        chunks = re.split("|".join(escaped), corpus)
    else:
        chunks = [corpus]

    corpus_freq: dict[tuple[bytes, ...], int] = {}

    for chunk in chunks:
        for match in re.finditer(PAT, chunk):
            pretok = match.group()
            pretok_bytes = pretok.encode("utf-8")

            # Represent every byte as a separate bytes object.
            key = tuple(bytes([b]) for b in pretok_bytes)

            # Count occurrences of this pre-token.
            corpus_freq[key] = corpus_freq.get(key, 0) + 1

    # Step 3: Learn BPE merges.
    # corpus_freq = {(b'c', b'a', b't'): 5}
    
    merges: list[tuple[bytes, bytes]] = []

    while len(vocab) < vocab_size:
        pair_freq: dict[tuple[bytes, bytes], int] = {}

        # Count adjacent pairs across all pre-tokens.
        for byte_tup, pretok_freq in corpus_freq.items():
            for i in range(len(byte_tup) - 1):
                pair = (byte_tup[i], byte_tup[i + 1])
                pair_freq[pair] = pair_freq.get(pair, 0) + pretok_freq

        # No adjacent pairs remain.
        if not pair_freq:
            break

        # Choose highest frequency, breaking ties lexicographically.
        max_byte_pair = max(
            pair_freq,
            key=lambda pair: (pair_freq[pair], pair)
        )

        # Record the merge and add the new token.
        merges.append(max_byte_pair)
        merged_bytes = max_byte_pair[0] + max_byte_pair[1]
        vocab[len(vocab)] = merged_bytes

        # Apply the merge to every pre-token.
        new_corpus_freq: dict[tuple[bytes, ...], int] = {}

        for byte_tup, pretok_freq in corpus_freq.items():
            merged_list: list[bytes] = []

            i = 0
            while i < len(byte_tup):
                if (i + 1 < len(byte_tup) and (byte_tup[i], byte_tup[i + 1]) == max_byte_pair):
                    merged_list.append(merged_bytes)
                    i += 2
                else:
                    merged_list.append(byte_tup[i])
                    i += 1

            merged_tup = tuple(merged_list)
            new_corpus_freq[merged_tup] = (
                new_corpus_freq.get(merged_tup, 0) + pretok_freq
            )

        corpus_freq = new_corpus_freq

    # Step 4: Return the vocabulary and merges.
    return vocab, merges

