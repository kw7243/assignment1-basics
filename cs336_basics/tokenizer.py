
from collections.abc import Iterable, Iterator
import os


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """
        Initialize a tokenizer using an existing vocabulary
        and learned BPE merge rules.
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

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        """
        Load a saved vocabulary and merge list.
        """
        # TODO: Deserialize vocab and merges.
        # Return an instance of cls.
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        """
        Convert text into a sequence of token IDs.
        """
        # TODO:
        # 1. Handle special tokens.
        # 2. Pre-tokenize.
        # 3. Convert pre-tokens into UTF-8 bytes.
        # 4. Apply learned BPE merges.
        # 5. Map final byte tokens to integer IDs.
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        """
        Convert token IDs into text.
        """
        # TODO: Reverse tokenization.
        raise NotImplementedError

    def encode_iterable(
        self, iterable: Iterable[str]
    ) -> Iterator[int]:
        """
        Lazily encode a stream of strings.
        """
        # TODO: Implement streaming encoding.
        raise NotImplementedError

