from collections.abc import Iterable, Iterator
import pickle
import os

import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
class Tokenizer:
    """
    A byte-level BPE tokenizer with an ordered list of merge rules.

    Token IDs map to unique byte strings. The vocabulary includes all 256
    single-byte values. Special tokens are matched as complete strings.
    """
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        """
        Initialize a tokenizer using an existing vocabulary
        and learned BPE merge rules.

        Args:
            vocab (dict[int, bytes]): Token bytes indexed by ID. Entries must be
                unique and include all 256 single-byte values.
            merges (list[tuple[bytes, bytes]]): Merge-priority order, earliest first.
            special_tokens (list[str] | None): Nonempty tokens to keep complete.
                Missing special tokens are added to a copy of the vocabulary.

        Raises:
            TypeError: If vocabulary entries have invalid types.
            ValueError: If token bytes are not unique or a base byte is missing.
        """

        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(special_tokens or [])

        # Reverse vocabulary: bytes -> token ID.
        self.byte_to_id = {
            token_bytes: token_id
            for token_id, token_bytes in self.vocab.items()
        }

        # Merge pair -> priority (lower means earlier).
        self.merge_ranks = {
            pair: rank
            for rank, pair in enumerate(self.merges)
        }

        # Register missing special tokens.
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")

            if token_bytes not in self.byte_to_id:
                token_id = max(self.vocab, default=-1) + 1
                self.vocab[token_id] = token_bytes
                self.byte_to_id[token_bytes] = token_id

        self._check_rep()

    def _check_rep(self) -> None:
        if len(self.byte_to_id) != len(self.vocab):
            raise ValueError("Vocabulary token bytes must be unique.")

        if not all(
            isinstance(token_id, int)
            and isinstance(token_bytes, bytes)
            for token_id, token_bytes in self.vocab.items()
        ):
            raise TypeError("Vocabulary entries must map integer token IDs to bytes.")

        if not all(
            bytes([i]) in self.byte_to_id
            for i in range(256)
        ):
            raise ValueError("Vocabulary must contain all 256 single-byte values.")

        if not all(
            self.byte_to_id[token_bytes] == token_id
            for token_id, token_bytes in self.vocab.items()
        ):
            raise RuntimeError("Vocabulary and reverse lookup are inconsistent.")
    
    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike[str],
        merges_filepath: str | os.PathLike[str],
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        """
        Load a saved vocabulary and merge list.

        Args:
            vocab_filepath (str | os.PathLike[str]): Binary pickle file containing a dict[int, bytes].
            merges_filepath (str | os.PathLike[str]): Binary pickle file containing a
                list[tuple[bytes, bytes]] in merge-priority order.
            special_tokens (list[str] | None): Additional complete tokens to register.

        Returns:
            Tokenizer: Uses the saved vocabulary and merge priorities.

        Requires:
            Both pickle files come from a trusted source. Loading a pickle
            can execute code.
        """
        # TODO: Deserialize vocab and merges.
        # Return an instance of cls.
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)

        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        Convert text into a sequence of token IDs.

        Registered special tokens remain complete. Ordinary text is encoded
        as UTF-8 bytes before the learned merge rules are applied.

        """
        # 1. Handle special tokens.
        # split by special tokens
        escaped = [
            re.escape(token)
            for token in sorted(set(self.special_tokens), key=len, reverse=True)

        ]
        chunks = (
            re.split("(" + "|".join(escaped) + ")", text)
            if escaped else [text]
        )

        # 2. Pre-tokenize. ["cat", "end", "dog"]
        special_token_set = set(self.special_tokens)
        token_ids: list[int] = []

        for chunk in chunks:
            if not chunk:
                continue


            if chunk in special_token_set:
                token_id = self.byte_to_id[chunk.encode("utf-8")]
                token_ids.append(token_id)
                continue

            # 3. Convert pre-tokens into UTF-8 bytes.
            for match in re.finditer(PAT, chunk):
                pretok_bytes = match.group().encode("utf-8")
                byte_tup = tuple(bytes([b]) for b in pretok_bytes)

                # pretokens = [(b"c", b"a", b't', b's'), (b"|<end>|", ), ('d', 'o', 'g')]
                # merges = [(b"c", b"a"), (b"ca", b"t"), (b"t", b"s")]
                
                # 4. Apply learned BPE merges if able
                while len(byte_tup) > 1:
                    mergeable_pairs = [(byte_tup[i], byte_tup[i+1]) 
                                    for i in range(len(byte_tup) - 1) 
                                    if (byte_tup[i], byte_tup[i+1]) in self.merge_ranks]

                    if not mergeable_pairs:
                        break

                    earliest_pair = min(mergeable_pairs, key=lambda pair: self.merge_ranks[pair])
                    
                    # Merge
                    merged_list: list[bytes] = []

                    i = 0
                    while i < len(byte_tup):
                        if (
                            i + 1 < len(byte_tup)
                            and (byte_tup[i], byte_tup[i + 1]) == earliest_pair
                        ):
                            earliest_pair_byte_str = earliest_pair[0] + earliest_pair[1]
                            merged_list.append(earliest_pair_byte_str)
                            i += 2
                        else:
                            merged_list.append(byte_tup[i])
                            i += 1

                    # 5. Map final byte tokens to integer IDs.
                    byte_tup = tuple(merged_list)

                for byte_str in byte_tup:
                    token_id = self.byte_to_id[byte_str]
                    token_ids.append(token_id)

        return token_ids


    def decode(self, ids: list[int]) -> str:
        """
        Convert token IDs into text.

        Concatenate the token bytes before decoding UTF-8. Invalid byte
        sequences are replaced with the Unicode replacement character.
        Each ID must be present in the vocabulary.

        """
        return b"".join([self.vocab[id] for id in ids]).decode("utf-8", errors="replace")

    def encode_iterable(
        self, iterable: Iterable[str]
    ) -> Iterator[int]:
        """
        Lazily encode a stream of strings.

        Treat the strings as consecutive chunks of one text stream. Special
        tokens and ordinary pre-tokens can span chunk boundaries. Yield
        token IDs and retain unfinished text between chunks.

        Args:
            iterable (Iterable[str]): Consecutive text chunks.

        Yields:
            int: Next token ID in the combined text stream.
        """
        

        special_tokens = sorted(
            set(self.special_tokens), key=len, reverse=True
        )
        special_token_set = set(special_tokens)

        if any(not token for token in special_tokens):
            raise ValueError("Special tokens must be nonempty")

        split_pattern = (
            re.compile("(" + "|".join(
                re.escape(token) for token in special_tokens
            ) + ")")
            if special_tokens else None
        )

        # Retain enough text to detect special tokens and
        # pre-tokens that span chunk boundaries.
        lookahead = max(
            (len(token) for token in special_tokens),
            default=0
        ) + 4

        buffer = ""

        for chunk in iterable:
            buffer += chunk

            cutoff = len(buffer) - lookahead
            if cutoff <= 0:
                continue

            parts = (
                split_pattern.split(buffer)
                if split_pattern else [buffer]
            )

            safe_end = 0
            offset = 0

            for part in parts:
                if not part:
                    continue

                if part in special_token_set:
                    # A complete special token is a safe boundary.
                    offset += len(part)
                    if offset <= cutoff:
                        safe_end = offset
                    continue

                # Find complete ordinary pre-tokens.
                for match in re.finditer(PAT, part):
                    end = offset + match.end()

                    if end > cutoff:
                        break

                    # Defer whitespace-ending matches because
                    # future text can change their boundaries.
                    if not match.group()[-1].isspace():
                        safe_end = end

                offset += len(part)

            # Encode only the stable prefix.
            if safe_end:
                yield from self.encode(buffer[:safe_end])
                buffer = buffer[safe_end:]

        # Process everything left at end of input.
        if buffer:
            yield from self.encode(buffer)
