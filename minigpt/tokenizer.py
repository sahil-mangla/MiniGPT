from abc import ABC, abstractmethod
from typing import Sequence


class Tokenizer(ABC):
    """
    Abstract Base Class defining the contract for all MiniGPT tokenizers.
    """

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Convert a string into a list of integer token IDs."""
        pass

    @abstractmethod
    def decode(self, tokens: Sequence[int]) -> str:
        """Convert a sequence of integer token IDs back into a string."""
        pass

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Return the total number of unique tokens in the vocabulary."""
        pass


class CharTokenizer(Tokenizer):
    """
    Character-level tokenizer.
    Constructs a deterministic vocabulary from unique characters in the provided text.
    """

    def __init__(self, text: str):
        if not text:
            raise ValueError("Training text for CharTokenizer cannot be empty.")

        # Extract unique characters and sort them deterministically
        self._chars: list[str] = sorted(list(set(text)))
        self._stoi: dict[str, int] = {ch: i for i, ch in enumerate(self._chars)}
        self._itos: dict[int, str] = {i: ch for i, ch in enumerate(self._chars)}

    @property
    def vocab_size(self) -> int:
        return len(self._chars)

    @property
    def vocabulary(self) -> list[str]:
        return list(self._chars)

    def encode(self, text: str) -> list[int]:
        """
        Encode a string into token IDs.
        Raises ValueError if an unknown character is encountered.
        """
        try:
            return [self._stoi[ch] for ch in text]
        except KeyError as e:
            unknown_char = e.args[0]
            raise ValueError(
                f"Character {repr(unknown_char)} not found in tokenizer vocabulary."
            ) from e

    def decode(self, tokens: Sequence[int]) -> str:
        """
        Decode a sequence of token IDs back into a string.
        Raises ValueError if an invalid token ID is provided.
        """
        try:
            return "".join([self._itos[t] for t in tokens])
        except KeyError as e:
            invalid_token = e.args[0]
            raise ValueError(
                f"Token ID {invalid_token} is outside vocabulary range [0, {self.vocab_size - 1}]."
            ) from e


def _get_pair_counts(ids: list[int]) -> dict[tuple[int, int], int]:
    """Count frequency of all consecutive pairs in a token ID list."""
    counts: dict[tuple[int, int], int] = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def _merge_pair(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace all consecutive occurrences of pair with new_id in ids."""
    new_ids = []
    i = 0
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(new_id)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


class BPETokenizer(Tokenizer):
    """
    Byte-Pair Encoding (BPE) Tokenizer from scratch (GPT-2 style).
    Operates directly on raw UTF-8 bytes to eliminate out-of-vocabulary tokens.
    """

    def __init__(
        self,
        merges: dict[tuple[int, int], int] | None = None,
        vocab: dict[int, bytes] | None = None,
    ):
        self.merges: dict[tuple[int, int], int] = merges or {}
        if vocab is not None:
            self.vocab: dict[int, bytes] = vocab
        else:
            # Base vocabulary: 256 raw bytes
            self.vocab = {idx: bytes([idx]) for idx in range(256)}
            for (p0, p1), new_id in self.merges.items():
                self.vocab[new_id] = self.vocab[p0] + self.vocab[p1]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @classmethod
    def train_from_text(
        cls, text: str, target_vocab_size: int = 300, verbose: bool = False
    ) -> "BPETokenizer":
        """
        Train a BPE tokenizer from raw text up to target_vocab_size.
        """
        if target_vocab_size < 256:
            raise ValueError(
                f"target_vocab_size must be >= 256 (base byte vocabulary), got {target_vocab_size}"
            )
        if not text:
            raise ValueError("Training text for BPETokenizer cannot be empty.")

        # Convert initial text to raw byte tokens (0..255)
        raw_bytes = list(text.encode("utf-8"))
        num_merges = target_vocab_size - 256

        ids = list(raw_bytes)
        merges: dict[tuple[int, int], int] = {}
        vocab: dict[int, bytes] = {idx: bytes([idx]) for idx in range(256)}

        for i in range(num_merges):
            stats = _get_pair_counts(ids)
            if not stats:
                break
            # Find the most frequent pair (break ties deterministically by pair content)
            best_pair = max(stats.keys(), key=lambda p: (stats[p], -p[0], -p[1]))
            if stats[best_pair] < 2:
                # No more recurring pairs to compress
                break

            new_id = 256 + i
            ids = _merge_pair(ids, best_pair, new_id)
            merges[best_pair] = new_id
            vocab[new_id] = vocab[best_pair[0]] + vocab[best_pair[1]]

            if verbose and (i + 1) % 50 == 0:
                print(f"Merge {i + 1}/{num_merges}: {best_pair} -> {new_id} ({vocab[new_id]!r})")

        return cls(merges=merges, vocab=vocab)

    def encode(self, text: str) -> list[int]:
        """
        Encode text into BPE token IDs by iteratively applying merges in rank order.
        """
        if not text:
            return []

        ids = list(text.encode("utf-8"))

        while len(ids) >= 2:
            stats = _get_pair_counts(ids)
            # Find the candidate pair in ids that was merged earliest
            candidate_pairs = [pair for pair in stats if pair in self.merges]
            if not candidate_pairs:
                break

            # Select pair with lowest merge index (highest merge priority)
            best_pair = min(candidate_pairs, key=lambda p: self.merges[p])
            new_id = self.merges[best_pair]
            ids = _merge_pair(ids, best_pair, new_id)

        return ids

    def decode(self, tokens: Sequence[int]) -> str:
        """
        Decode a sequence of BPE token IDs back into a UTF-8 string.
        """
        if not tokens:
            return ""

        byte_chunks = []
        for token_id in tokens:
            if token_id in self.vocab:
                byte_chunks.append(self.vocab[token_id])
            else:
                raise ValueError(
                    f"Token ID {token_id} is outside BPE vocabulary range [0, {self.vocab_size - 1}]."
                )

        raw_bytes = b"".join(byte_chunks)
        return raw_bytes.decode("utf-8", errors="replace")

    def compression_ratio(self, text: str) -> float:
        """
        Calculate compression ratio: len(raw_characters) / len(encoded_tokens).
        Higher is better (more text fits in context window).
        """
        encoded = self.encode(text)
        if not encoded:
            return 1.0
        return len(text) / len(encoded)
