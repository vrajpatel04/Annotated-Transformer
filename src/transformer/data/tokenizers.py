"""SpaCy tokenizers for Multi30k."""

from __future__ import annotations

import os

import spacy


class TokenizerManager:
    """Load or download German and English spaCy tokenizers."""

    def __init__(self, src_lang: str = "de", tgt_lang: str = "en"):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self._src_tokenizer = None
        self._tgt_tokenizer = None

    @staticmethod
    def _load_model(name: str):
        try:
            return spacy.load(name)
        except OSError:
            os.system(f"python -m spacy download {name}")
            return spacy.load(name)

    @property
    def src_tokenizer(self):
        if self._src_tokenizer is None:
            model = "de_core_news_sm" if self.src_lang == "de" else "en_core_web_sm"
            self._src_tokenizer = self._load_model(model)
        return self._src_tokenizer

    @property
    def tgt_tokenizer(self):
        if self._tgt_tokenizer is None:
            model = "en_core_web_sm" if self.tgt_lang == "en" else "de_core_news_sm"
            self._tgt_tokenizer = self._load_model(model)
        return self._tgt_tokenizer

    def tokenize_src(self, text: str) -> list[str]:
        return [tok.text for tok in self.src_tokenizer.tokenizer(text)]

    def tokenize_tgt(self, text: str) -> list[str]:
        return [tok.text for tok in self.tgt_tokenizer.tokenizer(text)]
