import os
import regex as re
from collections import Counter, defaultdict

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def count_pretokens(
    corpus: str,
    special_tokens: list[str],
) -> Counter[tuple[bytes, ...]]:

    escaped = [
        re.escape(token)
        for token in sorted(set(special_tokens), key=len, reverse=True)
    ]

    chunks = (
        re.split("|".join(escaped), corpus)
        if escaped else [corpus]
    )

    corpus_freq = Counter()

    for chunk in chunks:
        for match in re.finditer(PAT, chunk):
            pretok_bytes = match.group().encode("utf-8")
            byte_tup = tuple(bytes([b]) for b in pretok_bytes)

            corpus_freq[byte_tup] += 1

    return corpus_freq


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
    corpus_freq = kwargs.get("corpus_freq")

    if corpus_freq is None:
        with open(input_path, "r", encoding="utf-8") as f:
            corpus = f.read()

        corpus_freq = count_pretokens(corpus, special_tokens)


    # Step 3: Learn BPE merges.
    # corpus_freq = {(b'c', b'a', b't'): 5}
    merges: list[tuple[bytes, bytes]] = []

    # Give each pre-token a stable index.
    byte_tups = list(corpus_freq.keys())
    pretok_freqs = [corpus_freq[byte_tup] for byte_tup in byte_tups]

    # Initialize pair frequencies and reverse index.
    pair_freq: dict[tuple[bytes, bytes], int] = defaultdict(int)
    pair_to_words: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)

    for wid, byte_tup in enumerate(byte_tups):
        pretok_freq = pretok_freqs[wid]

        for byte_pair, count in Counter(
            zip(byte_tup, byte_tup[1:])
        ).items():
            pair_freq[byte_pair] += count * pretok_freq
            pair_to_words[byte_pair].add(wid)

    while len(vocab) < vocab_size:
        if not pair_freq:
            break

        # Find the most frequent pair.
        max_byte_pair = max(
            pair_freq,
            key=lambda pair: (pair_freq[pair], pair)
        )

        merges.append(max_byte_pair)
        max_byte_pair_str = max_byte_pair[0] + max_byte_pair[1]
        vocab[len(vocab)] = max_byte_pair_str

        # Only update pre-tokens containing the selected pair.
        affected_words = list(pair_to_words[max_byte_pair])

        for wid in affected_words:
            byte_tup = byte_tups[wid]
            pretok_freq = pretok_freqs[wid]

            # Remove old pair frequencies.
            old_pairs = Counter(zip(byte_tup, byte_tup[1:]))

            for byte_pair, count in old_pairs.items():
                pair_freq[byte_pair] -= count * pretok_freq

                if pair_freq[byte_pair] == 0:
                    del pair_freq[byte_pair]

                pair_to_words[byte_pair].remove(wid)

                if not pair_to_words[byte_pair]:
                    del pair_to_words[byte_pair]

            # Merge selected pair.
            merged_list: list[bytes] = []

            i = 0
            while i < len(byte_tup):
                if (
                    i + 1 < len(byte_tup)
                    and (byte_tup[i], byte_tup[i + 1]) == max_byte_pair
                ):
                    merged_list.append(max_byte_pair_str)
                    i += 2
                else:
                    merged_list.append(byte_tup[i])
                    i += 1

            merged_tup = tuple(merged_list)
            byte_tups[wid] = merged_tup

            # Add new pair frequencies.
            new_pairs = Counter(zip(merged_tup, merged_tup[1:]))

            for byte_pair, count in new_pairs.items():
                pair_freq[byte_pair] += count * pretok_freq
                pair_to_words[byte_pair].add(wid)

    return vocab, merges


