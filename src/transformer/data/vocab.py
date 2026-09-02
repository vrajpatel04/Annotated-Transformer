"""Simple vocabulary compatible with the former torchtext Vocab API."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator


class Vocab:
    """String-to-index vocabulary with optional unknown-token fallback."""

    def __init__(
        self,
        stoi: dict[str, int],
        itos: list[str] | None = None,
        default_index: int | None = None,
    ):
        self._stoi = stoi
        self._itos = itos if itos is not None else [None] * len(stoi)
        if itos is None:
            for token, index in stoi.items():
                self._itos[index] = token
        self._default_index = default_index

    def __len__(self) -> int:
        return len(self._itos)

    def __getitem__(self, token: str) -> int:
        if token in self._stoi:
            return self._stoi[token]
        if self._default_index is not None:
            return self._default_index
        raise KeyError(token)

    def __call__(self, tokens: Iterable[str]) -> list[int]:
        return [self[token] for token in tokens]

    def get_stoi(self) -> dict[str, int]:
        return self._stoi

    def get_itos(self) -> list[str]:
        return self._itos

    def set_default_index(self, index: int) -> None:
        self._default_index = index


def build_vocab_from_iterator(
    token_iterator: Iterator[Iterable[str]],
    *,
    min_freq: int = 1,
    specials: list[str] | None = None,
) -> Vocab:
    """Build a vocabulary from token streams."""
    counter: Counter[str] = Counter()
    for tokens in token_iterator:
        counter.update(tokens)

    specials = specials or []
    itos = list(specials)
    for token, freq in counter.items():
        if freq >= min_freq and token not in itos:
            itos.append(token)

    stoi = {token: index for index, token in enumerate(itos)}
    return Vocab(stoi, itos)
