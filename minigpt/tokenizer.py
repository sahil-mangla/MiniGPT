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
