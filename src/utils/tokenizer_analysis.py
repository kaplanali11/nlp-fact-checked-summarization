"""
HAFTA 7 - K1: SUBWORD TOKENIZER ANALİZİ VE UZUNLUK PARAMETRESİ SEÇİMİ

Temizlenmiş TR-News dosyaları (data/train_clean.csv, val_clean.csv, test_clean.csv)
üzerinde, seçilen modelin GERÇEK tokenizer'ı ile şunları ölçer:

  1) Kaynak ve özet token uzunluk yüzdelikleri, kelime -> token genişleme çarpanı
  2) Aday max_input_length / max_target_length değerleri için:
       - kapsama        : o uzunluğa tamamen sığan (kırpılmayan) doküman yüzdesi
       - token koruması : toplam token'ların kırpma sonrası korunan yüzdesi
       - ort. kullanılan token : dinamik padding ile batch başına gerçek yük
  3) Tokenizer sağlık kontrolleri:
       - <unk> oranı, Türkçe harflerin (çğıöşü İ) gidiş-dönüş testi
       - özel token kimlikleri (pad/eos/unk), sözlük boyutu
       - isteğe bağlı kaynak ön ekinin ("summarize: " gibi) token maliyeti

NOT: Önceden eğitilmiş modelin sözlüğü değiştirilemez; "tokenizer parametresi
ayarı" burada uzunluk limitleri, truncation ve padding kararlarıdır.
Sonuçlar YALNIZCA --model ile verilen tokenizer için geçerlidir. K3 başka bir
model seçerse bu betik o model ile yeniden çalıştırılmalıdır.

Çalıştırma:
  python tokenizer_analysis.py
  python tokenizer_analysis.py --model savasy/mt5-mlsum-turkish-summarization
  python tokenizer_analysis.py --splits train --sample 50000      # hızlı deneme
  python tokenizer_analysis.py --source-prefix "summarize: "
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

DEFAULT_MODEL = "savasy/mt5-mlsum-turkish-summarization"
SPLIT_FILES = {"train": "train_clean.csv", "val": "val_clean.csv", "test": "test_clean.csv"}

INPUT_LIMITS = [256, 384, 512, 768, 1024, 1536, 2048]
TARGET_LIMITS = [32, 48, 64, 96, 128, 150]
PERCENTILES = (50, 90, 95, 98, 99)
TURKISH_CHARS = "çğıöşüÇĞİÖŞÜâîû"


def load_split(path: Path, sample: int | None, seed: int) -> pd.DataFrame:
    # keep_default_na=False: "NA", "null" gibi metinler NaN'a çevrilmesin
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in ("clean_text", "clean_summary"):
        if col not in df.columns:
            raise ValueError(f"{path} içinde '{col}' sütunu yok. Sütunlar: {list(df.columns)}")
    if sample and sample < len(df):
        df = df.sample(n=sample, random_state=seed)
    return df.reset_index(drop=True)


def token_lengths(tokenizer, texts: Sequence[str], batch_size: int) -> np.ndarray:
    """Kırpma yapmadan, özel tokenlar (EOS) dahil gerçek token uzunlukları."""
    lengths: List[int] = []
    for start in range(0, len(texts), batch_size):
        enc = tokenizer(
            list(texts[start:start + batch_size]),
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )
        lengths.extend(len(ids) for ids in enc["input_ids"])
    return np.asarray(lengths, dtype=np.int64)


def truncation_table(lengths: np.ndarray, limits: Sequence[int]) -> pd.DataFrame:
    rows = []
    total = lengths.sum()
    for limit in limits:
        kept = np.minimum(lengths, limit)
        rows.append(
            {
                "limit": limit,
                "coverage_pct": float((lengths <= limit).mean() * 100),
                "truncated_docs_pct": float((lengths > limit).mean() * 100),
                "token_retention_pct": float(kept.sum() / total * 100),
                "mean_tokens_used": float(kept.mean()),
            }
        )
    return pd.DataFrame(rows)


def first_limit_reaching(table: pd.DataFrame, column: str, threshold: float):
    hit = table[table[column] >= threshold]
    return int(hit.iloc[0]["limit"]) if len(hit) else None


def health_checks(tokenizer, texts: Sequence[str], n_unk_docs: int, n_roundtrip: int) -> Dict[str, object]:
    out: Dict[str, object] = {}

    unk_id = tokenizer.unk_token_id
    total_tokens = unk_tokens = 0
    for text in texts[:n_unk_docs]:
        ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
        total_tokens += len(ids)
        if unk_id is not None:
            unk_tokens += ids.count(unk_id)
    out["unk_rate_pct"] = (unk_tokens / total_tokens * 100) if total_tokens else 0.0
    out["unk_check_docs"] = min(n_unk_docs, len(texts))

    char_failures = []
    for ch in TURKISH_CHARS:
        ids = tokenizer(ch, add_special_tokens=False)["input_ids"]
        back = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if back.strip() != ch:
            char_failures.append((ch, back))
    out["char_failures"] = char_failures

    exact = 0
    examples = []
    checked = texts[:n_roundtrip]
    for text in checked:
        ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
        back = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if back == text:
            exact += 1
        elif len(examples) < 3:
            examples.append((text[:90], back[:90]))
    out["roundtrip_exact_pct"] = exact / max(len(checked), 1) * 100
    out["roundtrip_examples"] = examples
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Hafta 7 K1: tokenizer analizi")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="eda_sonuclari")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=list(SPLIT_FILES))
    parser.add_argument("--sample", type=int, default=None, help="Hızlı deneme için rastgele N doküman")
    parser.add_argument("--batch-size", type=int, default=256, help="Tokenizer batch boyutu")
    parser.add_argument("--source-prefix", default="", help='Örn. "summarize: " (modelin beklediği ön ek)')
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 84)
    print("HAFTA 7: K1 SUBWORD TOKENIZER ANALİZİ")
    print("=" * 84)
    print(f"Model/tokenizer : {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    print(f"Fast tokenizer  : {tokenizer.is_fast}")
    print(f"Sözlük boyutu   : {len(tokenizer):,}")
    print(f"pad / eos / unk : {tokenizer.pad_token_id} / {tokenizer.eos_token_id} / {tokenizer.unk_token_id}")
    print(f"model_max_length: {tokenizer.model_max_length}  (limit vermeden truncation=True bu değere kırpar)")

    prefix_tokens = 0
    if args.source_prefix:
        prefix_tokens = len(tokenizer(args.source_prefix, add_special_tokens=False)["input_ids"])
        print(f"Kaynak ön eki   : {args.source_prefix!r} -> {prefix_tokens} token (girdi bütçesinden düşer)")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.model.strip("/\\").split("/")[-1])

    trunc_rows: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, object]] = []
    train_texts: List[str] = []

    for split in args.splits:
        path = Path(args.data_dir) / SPLIT_FILES[split]
        print("\n" + "-" * 84)
        print(f"SPLIT: {split}  ({path})")
        print("-" * 84)
        df = load_split(path, args.sample, args.seed)
        texts = (args.source_prefix + df["clean_text"]).tolist()
        summaries = df["clean_summary"].tolist()
        if split == "train":
            train_texts = texts

        start = time.time()
        src_len = token_lengths(tokenizer, texts, args.batch_size)
        tgt_len = token_lengths(tokenizer, summaries, args.batch_size)
        print(f"{len(df):,} doküman tokenize edildi ({time.time() - start:.0f} sn)")

        word_total = df["clean_text"].str.split().str.len().sum()
        # Ön ek token'ları oranı şişirmesin diye çıkarılır
        ratio = (src_len.sum() - prefix_tokens * len(src_len)) / max(word_total, 1)
        src_p = np.percentile(src_len, PERCENTILES).astype(int)
        tgt_p = np.percentile(tgt_len, PERCENTILES).astype(int)

        print(f"Token/kelime çarpanı (kaynak): {ratio:.3f}")
        print(f"{'Yüzdelik':<10}{'Kaynak token':>14}{'Özet token':>12}")
        for p, s, t in zip(PERCENTILES, src_p, tgt_p):
            print(f"%{p:<9}{s:>14,}{t:>12,}")

        t_in = truncation_table(src_len, INPUT_LIMITS)
        t_tg = truncation_table(tgt_len, TARGET_LIMITS)
        for name, table in (("max_input_length", t_in), ("max_target_length", t_tg)):
            print(f"\n{name}  (kapsama = kırpılmayan doküman %, koruma = korunan token %)")
            print(f"{'limit':>7}{'kapsama':>10}{'kırpılan':>10}{'koruma':>9}{'ort.token':>11}")
            for _, r in table.iterrows():
                print(f"{int(r['limit']):>7}{r['coverage_pct']:>9.2f}%{r['truncated_docs_pct']:>9.2f}%"
                      f"{r['token_retention_pct']:>8.2f}%{r['mean_tokens_used']:>11.1f}")

        for kind, table in (("source", t_in), ("target", t_tg)):
            t = table.copy()
            t.insert(0, "kind", kind)
            t.insert(0, "split", split)
            trunc_rows.append(t)

        summary_row = {
            "split": split, "docs": len(df), "tokens_per_word": ratio,
            "input_limit_for_95pct_coverage": first_limit_reaching(t_in, "coverage_pct", 95),
            "input_limit_for_98pct_retention": first_limit_reaching(t_in, "token_retention_pct", 98),
            "target_limit_for_99pct_coverage": first_limit_reaching(t_tg, "coverage_pct", 99),
        }
        for p, s, t in zip(PERCENTILES, src_p, tgt_p):
            summary_row[f"source_tokens_p{p}"] = int(s)
            summary_row[f"summary_tokens_p{p}"] = int(t)
        summary_rows.append(summary_row)

        print(f"\n-> %95 kapsama için en küçük max_input_length : {summary_row['input_limit_for_95pct_coverage']}")
        print(f"-> %98 token koruması için en küçük max_input_length: {summary_row['input_limit_for_98pct_retention']}")
        print(f"-> %99 kapsama için en küçük max_target_length: {summary_row['target_limit_for_99pct_coverage']}")

    if train_texts:
        print("\n" + "=" * 84)
        print("TOKENIZER SAĞLIK KONTROLLERİ (train üzerinden)")
        print("=" * 84)
        checks = health_checks(tokenizer, train_texts, n_unk_docs=20000, n_roundtrip=1000)
        print(f"<unk> oranı ({checks['unk_check_docs']:,} doküman)       : %{checks['unk_rate_pct']:.4f}")
        if checks["char_failures"]:
            print("Türkçe harf gidiş-dönüş HATALARI:")
            for ch, back in checks["char_failures"]:
                print(f"   {ch!r} -> {back!r}")
        else:
            print(f"Türkçe harfler ({TURKISH_CHARS}) gidiş-dönüşte korunuyor.")
        print(f"decode(encode(x)) == x olan metin       : %{checks['roundtrip_exact_pct']:.1f}")
        print("(Tam eşleşmeme normalizasyondan (NFKC, boşluk) gelebilir; harf kaybı değildir.)")
        for original, back in checks["roundtrip_examples"]:
            print(f"   orijinal: {original}\n   geri     : {back}")

    pd.concat(trunc_rows).to_csv(out_dir / f"tokenizer_kirpma_{slug}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary_rows).to_csv(out_dir / f"tokenizer_ozet_{slug}.csv", index=False, encoding="utf-8-sig")
    print(f"\nCSV çıktıları: {out_dir.resolve()}")
    print("\nKARAR İÇİN: Kapsama, kırpılmayan doküman oranıdır. Koruma ise kırpmada kaybedilen "
          "token payını gösterir; haberlerde önemli bilgi genelde metnin başında toplandığı için "
          "koruma, kapsamadan daha yumuşak bir ölçüttür. Nihai seçimi GPU belleği ile birlikte verin.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
