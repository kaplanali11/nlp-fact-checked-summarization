"""
HAFTA 7 - K1: SEQ2SEQ ÖZETLEME İÇİN DATALOADER (TR-News)

Girdi : data/train_clean.csv, val_clean.csv, test_clean.csv
        (sütunlar: clean_text, clean_summary; cleaning.py çıktısı)
Çıktı : PyTorch DataLoader'lar; her batch şu anahtarları içerir:
          input_ids, attention_mask, labels
        labels içindeki padding -100'dür (loss hesabında yok sayılır).
        decoder_input_ids VERİLMEZ: T5/mT5 modelleri labels verildiğinde
        bunları kendisi üretir (sağa kaydırma).

Tasarım kararları:
  - Tokenizasyon collate aşamasında, batch başına yapılır; 276 bin dokümanı
    önceden tokenize edip bellekte tutmak gerekmez.
  - Padding batch içindeki en uzun örneğe göre (dinamik) yapılır ve 8'in katına
    tamamlanır (GPU verimliliği).
  - group_by_length=True: train'de benzer uzunluktaki örnekler aynı batch'e
    düşer, padding israfı azalır. İlk batch bilerek en uzun olandır; bellek
    yetmeyecekse eğitimin başında patlar, saatler sonra değil.
  - Ön ek (source_prefix) modele bağlıdır. mT5 için gerekmeyebilir, orijinal
    T5 için "summarize: " kullanılır. Modelin kartından doğrulayın.

Hızlı deneme (veri ve model gerekir):
  python dataloader.py --model savasy/mt5-mlsum-turkish-summarization --limit 2000
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

# Windows/çok işçili kullanımda fast tokenizer uyarılarını önler
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SPLIT_FILES = {"train": "train_clean.csv", "val": "val_clean.csv", "test": "test_clean.csv"}
DEFAULT_MODEL = "savasy/mt5-mlsum-turkish-summarization"


# -----------------------------------------------------------------------------
# DATASET
# -----------------------------------------------------------------------------
class SummarizationDataset(Dataset):
    """clean_text -> clean_summary çiftlerini ham metin olarak tutar."""

    def __init__(self, csv_path: Path | str, source_prefix: str = "", limit: Optional[int] = None):
        # keep_default_na=False: "NA", "null" gibi metinler NaN'a çevrilmesin
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        missing = {"clean_text", "clean_summary"} - set(df.columns)
        if missing:
            raise ValueError(f"{csv_path} içinde eksik sütunlar: {sorted(missing)}")
        if limit:
            df = df.head(limit)

        self.texts: List[str] = (source_prefix + df["clean_text"]).tolist()
        self.summaries: List[str] = df["clean_summary"].tolist()
        # Gruplama için ucuz uzunluk göstergesi (kelime sayısı)
        self.lengths: List[int] = [len(t.split()) for t in self.texts]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return {"text": self.texts[idx], "summary": self.summaries[idx]}


# -----------------------------------------------------------------------------
# COLLATOR
# -----------------------------------------------------------------------------
@dataclass
class Seq2SeqCollator:
    """Batch'i tokenize eder, dinamik padding uygular, label padding'ini -100 yapar."""

    tokenizer: object
    max_input_length: int = 512
    max_target_length: int = 128
    pad_to_multiple_of: Optional[int] = 8
    label_pad_token_id: int = -100

    def __post_init__(self) -> None:
        # Tokenizer, kırpma uzunluğu pad_to_multiple_of'un katı değilse hata verir
        # (ör. max_input_length=1000). Bu durumda ilgili taraf için 8'lik hizalama
        # kapatılır; dinamik padding yine çalışır.
        m = self.pad_to_multiple_of
        self._in_multiple = m if m and self.max_input_length % m == 0 else None
        self._tgt_multiple = m if m and self.max_target_length % m == 0 else None

    def __call__(self, batch: Sequence[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            [item["text"] for item in batch],
            max_length=self.max_input_length,
            truncation=True,
            padding=True,
            pad_to_multiple_of=self._in_multiple,
            return_tensors="pt",
        )
        tgt = self.tokenizer(
            text_target=[item["summary"] for item in batch],
            max_length=self.max_target_length,
            truncation=True,
            padding=True,
            pad_to_multiple_of=self._tgt_multiple,
            return_tensors="pt",
        )
        # Padding konumları attention_mask'ten alınır (pad kimliğine güvenilmez)
        labels = tgt["input_ids"].clone()
        labels[tgt["attention_mask"] == 0] = self.label_pad_token_id
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
        }


# -----------------------------------------------------------------------------
# UZUNLUĞA GÖRE GRUPLAYAN BATCH SAMPLER
# -----------------------------------------------------------------------------
class LengthGroupedBatchSampler(Sampler[List[int]]):
    """
    Örnekleri karıştırır, büyük parçalara (mega-batch) böler, her parçayı uzunluğa
    göre sıralayıp batch'ler. Batch'ler yeniden karıştırılır; en uzun batch başa alınır.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        seed: int = 42,
        mega_batch_mult: int = 50,
        longest_first: bool = True,
        drop_last: bool = False,
    ):
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.seed = seed
        self.mega_batch_mult = mega_batch_mult
        self.longest_first = longest_first
        self.drop_last = drop_last
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        n = len(self.lengths)
        return n // self.batch_size if self.drop_last else math.ceil(n / self.batch_size)

    def __iter__(self) -> Iterator[List[int]]:
        gen = torch.Generator()
        gen.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.lengths), generator=gen).tolist()

        mega = self.batch_size * self.mega_batch_mult
        batches: List[List[int]] = []
        for start in range(0, len(indices), mega):
            chunk = sorted(indices[start:start + mega], key=lambda i: self.lengths[i], reverse=True)
            for k in range(0, len(chunk), self.batch_size):
                batch = chunk[k:k + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)

        order = torch.randperm(len(batches), generator=gen).tolist()
        batches = [batches[i] for i in order]

        if self.longest_first and batches:
            longest = max(range(len(batches)), key=lambda i: sum(self.lengths[j] for j in batches[i]))
            batches.insert(0, batches.pop(longest))

        yield from batches


# -----------------------------------------------------------------------------
# DATALOADER OLUŞTURUCU
# -----------------------------------------------------------------------------
def build_dataloaders(
    tokenizer,
    data_dir: str = "data",
    batch_size: int = 8,
    eval_batch_size: Optional[int] = None,
    max_input_length: int = 512,
    max_target_length: int = 128,
    num_workers: int = 0,
    group_by_length: bool = True,
    source_prefix: str = "",
    seed: int = 42,
    splits: Sequence[str] = ("train", "val", "test"),
    limit: Optional[int] = None,
) -> Dict[str, DataLoader]:
    """
    Her split için DataLoader döndürür. Train: karıştırılır (isteğe bağlı uzunluk
    gruplamasıyla). Val/test: sıra korunur, karıştırılmaz.

    num_workers: Windows'ta varsayılan 0 bırakın; >0 için betiğin
    `if __name__ == "__main__":` içinde çalışması gerekir.
    """
    collator = Seq2SeqCollator(tokenizer, max_input_length, max_target_length)
    loaders: Dict[str, DataLoader] = {}

    for split in splits:
        dataset = SummarizationDataset(Path(data_dir) / SPLIT_FILES[split], source_prefix, limit)
        if split == "train":
            if group_by_length:
                sampler = LengthGroupedBatchSampler(dataset.lengths, batch_size, seed=seed)
                loaders[split] = DataLoader(
                    dataset, batch_sampler=sampler, collate_fn=collator, num_workers=num_workers
                )
            else:
                gen = torch.Generator()
                gen.manual_seed(seed)
                loaders[split] = DataLoader(
                    dataset, batch_size=batch_size, shuffle=True, generator=gen,
                    collate_fn=collator, num_workers=num_workers,
                )
        else:
            loaders[split] = DataLoader(
                dataset, batch_size=eval_batch_size or batch_size * 2, shuffle=False,
                collate_fn=collator, num_workers=num_workers,
            )
    return loaders


# -----------------------------------------------------------------------------
# DUMAN TESTİ
# -----------------------------------------------------------------------------
def padding_ratio(loader: DataLoader, n_batches: int) -> float:
    pad = total = 0
    for batch in islice(loader, n_batches):
        mask = batch["attention_mask"]
        pad += int((mask == 0).sum())
        total += mask.numel()
    return pad / max(total, 1) * 100


def smoke_test(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    common = dict(
        data_dir=args.data_dir, batch_size=args.batch_size,
        max_input_length=args.max_input_length, max_target_length=args.max_target_length,
        source_prefix=args.source_prefix, limit=args.limit,
    )
    loaders = build_dataloaders(tokenizer, group_by_length=True, **common)
    for name, loader in loaders.items():
        print(f"{name:<6}: {len(loader.dataset):,} örnek, {len(loader):,} batch")

    batch = next(iter(loaders["train"]))
    print("\nİlk train batch (en uzun batch'tir):")
    for key, value in batch.items():
        print(f"  {key:<15}: {tuple(value.shape)} {value.dtype}")

    eos = tokenizer.eos_token_id
    for key in ("input_ids", "labels"):
        ids = batch[key]
        mask = batch["attention_mask"] if key == "input_ids" else (ids != -100)
        last = ids.gather(1, (mask.sum(1) - 1).unsqueeze(1)).squeeze(1)
        ok = bool((last == eos).all())
        print(f"  {key:<15}: her satır EOS ile bitiyor mu? {'EVET' if ok else 'HAYIR (kontrol edin!)'}")

    first_label = batch["labels"][0]
    print("\nİlk örnek (kaynak, ilk 200 karakter):")
    print("  ", tokenizer.decode(batch["input_ids"][0], skip_special_tokens=True)[:200])
    print("İlk örnek (özet):")
    print("  ", tokenizer.decode(first_label[first_label != -100], skip_special_tokens=True))

    n = args.padding_batches
    grouped = padding_ratio(loaders["train"], n)
    plain = padding_ratio(
        build_dataloaders(tokenizer, group_by_length=False, splits=("train",), **common)["train"], n
    )
    print(f"\nPadding oranı ({n} batch): gruplamasız %{plain:.1f} | uzunluğa göre gruplu %{grouped:.1f}")

    start = time.time()
    count = sum(1 for _ in islice(loaders["train"], n))
    elapsed = time.time() - start
    print(f"Veri hazırlama hızı: {count / elapsed:.1f} batch/sn (yalnızca veri yükleme, model yok)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hafta 7 K1: DataLoader duman testi")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--max-target-length", type=int, default=128)
    parser.add_argument("--source-prefix", default="")
    parser.add_argument("--limit", type=int, default=None, help="Her split'ten ilk N satır (hızlı deneme)")
    parser.add_argument("--padding-batches", type=int, default=50)
    smoke_test(parser.parse_args())


if __name__ == "__main__":
    main()
