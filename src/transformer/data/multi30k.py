"""Multi30k dataset loading, vocabulary, and dataloaders."""

from __future__ import annotations

import torch
from datasets import load_dataset
from torch.nn.functional import pad
from torch.utils.data import DataLoader, Dataset

from transformer.config import TrainingConfig
from transformer.data.tokenizers import TokenizerManager
from transformer.data.vocab import Vocab, build_vocab_from_iterator


SPECIALS = ["<s>", "</s>", "<blank>", "<unk>"]
HF_DATASET_NAME = "bentrevett/multi30k"
HF_SPLIT_NAMES = {
    "train": "train",
    "valid": "validation",
    "test": "test",
}

_SPLIT_CACHE: dict[tuple[str, str], tuple["_Multi30kSplit", "_Multi30kSplit", "_Multi30kSplit"]] = {}


class _Multi30kSplit(Dataset):
    """Map-style Multi30k split as (source, target) sentence pairs."""

    def __init__(self, pairs: list[tuple[str, str]]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[str, str]:
        return self.pairs[index]


def _pairs_from_hf_split(
    split: str,
    src_lang: str,
    tgt_lang: str,
    dataset_dict,
) -> list[tuple[str, str]]:
    rows = dataset_dict[HF_SPLIT_NAMES[split]]
    return [(row[src_lang], row[tgt_lang]) for row in rows]


def load_multi30k_splits(
    language_pair: tuple[str, str],
) -> tuple[_Multi30kSplit, _Multi30kSplit, _Multi30kSplit]:
    """Load train/valid/test Multi30k splits from Hugging Face."""
    if language_pair in _SPLIT_CACHE:
        return _SPLIT_CACHE[language_pair]

    src_lang, tgt_lang = language_pair
    print(f"Loading {HF_DATASET_NAME} from Hugging Face ...")
    dataset_dict = load_dataset(HF_DATASET_NAME)

    splits = (
        _Multi30kSplit(_pairs_from_hf_split("train", src_lang, tgt_lang, dataset_dict)),
        _Multi30kSplit(_pairs_from_hf_split("valid", src_lang, tgt_lang, dataset_dict)),
        _Multi30kSplit(_pairs_from_hf_split("test", src_lang, tgt_lang, dataset_dict)),
    )
    _SPLIT_CACHE[language_pair] = splits
    return splits


class Multi30kDataModule:
    """Prepare Multi30k vocabularies and PyTorch dataloaders."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        src_lang, tgt_lang = config.language_pair
        self.tokenizers = TokenizerManager(src_lang, tgt_lang)
        self.vocab_src: Vocab | None = None
        self.vocab_tgt: Vocab | None = None
        self.pad_idx: int | None = None

    @staticmethod
    def _yield_tokens(data_iter, tokenizer_fn, index: int):
        for sample in data_iter:
            yield tokenizer_fn(sample[index])

    def build_vocabularies(self) -> tuple[Vocab, Vocab]:
        if self.config.vocab_path.exists():
            self.vocab_src, self.vocab_tgt = torch.load(
                self.config.vocab_path, weights_only=False
            )
        else:
            train, valid, test = load_multi30k_splits(self.config.language_pair)
            combined = train.pairs + valid.pairs + test.pairs

            print("Building source vocabulary ...")
            self.vocab_src = build_vocab_from_iterator(
                self._yield_tokens(
                    combined, self.tokenizers.tokenize_src, index=0
                ),
                min_freq=self.config.vocab_min_freq,
                specials=SPECIALS,
            )

            print("Building target vocabulary ...")
            self.vocab_tgt = build_vocab_from_iterator(
                self._yield_tokens(
                    combined, self.tokenizers.tokenize_tgt, index=1
                ),
                min_freq=self.config.vocab_min_freq,
                specials=SPECIALS,
            )

            self.vocab_src.set_default_index(self.vocab_src["<unk>"])
            self.vocab_tgt.set_default_index(self.vocab_tgt["<unk>"])

            self.config.vocab_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save((self.vocab_src, self.vocab_tgt), self.config.vocab_path)

        self.pad_idx = self.vocab_tgt["<blank>"]
        print(
            f"Vocabulary sizes — source: {len(self.vocab_src)}, "
            f"target: {len(self.vocab_tgt)}"
        )
        return self.vocab_src, self.vocab_tgt

    def _collate_batch(
        self,
        batch,
        device: torch.device,
        max_padding: int,
        pad_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bs_id = torch.tensor([0], device=device)
        eos_id = torch.tensor([1], device=device)
        src_list, tgt_list = [], []

        for src_text, tgt_text in batch:
            processed_src = torch.cat(
                [
                    bs_id,
                    torch.tensor(
                        self.vocab_src(self.tokenizers.tokenize_src(src_text)),
                        dtype=torch.int64,
                        device=device,
                    ),
                    eos_id,
                ],
                0,
            )
            processed_tgt = torch.cat(
                [
                    bs_id,
                    torch.tensor(
                        self.vocab_tgt(self.tokenizers.tokenize_tgt(tgt_text)),
                        dtype=torch.int64,
                        device=device,
                    ),
                    eos_id,
                ],
                0,
            )
            src_list.append(
                pad(
                    processed_src,
                    (0, max_padding - len(processed_src)),
                    value=pad_id,
                )
            )
            tgt_list.append(
                pad(
                    processed_tgt,
                    (0, max_padding - len(processed_tgt)),
                    value=pad_id,
                )
            )

        return torch.stack(src_list), torch.stack(tgt_list)

    def create_dataloaders(
        self,
        device: torch.device,
        batch_size: int | None = None,
        include_test: bool = False,
    ) -> tuple[DataLoader, DataLoader] | tuple[DataLoader, DataLoader, DataLoader]:
        if self.vocab_src is None or self.vocab_tgt is None:
            self.build_vocabularies()

        batch_size = batch_size or self.config.batch_size
        pad_id = self.vocab_src.get_stoi()["<blank>"]

        def collate_fn(batch):
            return self._collate_batch(
                batch,
                device,
                self.config.max_padding,
                pad_id,
            )

        train_map, valid_map, test_map = load_multi30k_splits(
            self.config.language_pair
        )

        train_loader = DataLoader(
            train_map,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        valid_loader = DataLoader(
            valid_map,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        if include_test:
            test_loader = DataLoader(
                test_map,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
            )
            return train_loader, valid_loader, test_loader
        return train_loader, valid_loader

    def ids_to_tokens(self, ids: torch.Tensor, side: str = "tgt") -> list[str]:
        vocab = self.vocab_tgt if side == "tgt" else self.vocab_src
        itos = vocab.get_itos()
        return [itos[i] for i in ids.tolist() if i != self.pad_idx]

    def decode_prediction(self, ids: torch.Tensor, eos: str = "</s>") -> str:
        tokens = self.ids_to_tokens(ids, side="tgt")
        text = " ".join(tokens)
        return text.split(eos, 1)[0] + eos if eos in text else text
