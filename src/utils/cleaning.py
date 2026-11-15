"""
HAFTA 6 - K1: TR-News VERİ TEMİZLEME BORU HATTI

TR-News (batubayk/tr-news) zaten train / validation / test olarak bölünmüş gelir.
Bu betik veriyi YENİDEN BÖLMEZ; üç resmi split'i ayrı ayrı temizler:

  1) Metin temizliği: HTML etiketi, URL, e-posta, fazla boşluk
  2) Bozuk kayıt filtresi:
       - boş kaynak/özet                     -> tüm split'lerde elenir
       - kaynak < 15 kelime, özet < 3 kelime -> train ve validation'da elenir
         (test'e uygulanmaz; --filter-test ile açılabilir)
       - kaynak > N kelime (isteğe bağlı)    -> yalnızca train'de, --max-source-words ile
  3) Tekrar ve sızıntı kontrolü:
       - train içindeki aynı kaynak metinler (ilki kalır)
       - train'de olup validation/test'te de bulunan metinler (train'den çıkarılır)
       - validation'da olup test'te de bulunan metinler (validation'dan çıkarılır)
       - validation/test içindeki kendi tekrarları yalnızca raporlanır, silinmez

Çıktılar (varsayılan: data/):
  train_clean.csv, val_clean.csv, test_clean.csv   (sütunlar: clean_text, clean_summary)
  cleaning_istatistikleri.csv                      (rapor için adım adım sayılar)

Çalıştırma:
  python cleaning.py
  python cleaning.py --max-source-words 1200      # isteğe bağlı uzunluk filtresi (yalnızca train)
  python cleaning.py --output-dir data --filter-test
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

# -----------------------------------------------------------------------------
# REGEX'LER
# -----------------------------------------------------------------------------
# Yalnızca gerçek etiketleri hedefler ("a < b" gibi ifadeleri silmez).
HTML_RE = re.compile(r"</?[A-Za-z][^>]*>")
URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
WS_RE = re.compile(r"\s+")

SPLIT_OUTPUT_NAMES = {
    "train": "train_clean.csv",
    "validation": "val_clean.csv",
    "test": "test_clean.csv",
}


def clean_text(text: object) -> str:
    """HTML etiketlerini, URL'leri, e-postaları ve fazla boşlukları temizler."""
    if not isinstance(text, str):
        return ""
    text = HTML_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    return WS_RE.sub(" ", text).strip()


def has_pattern(value: object, pattern: re.Pattern) -> bool:
    return isinstance(value, str) and bool(pattern.search(value))


def find_column(columns, candidates) -> Optional[str]:
    lookup = {str(c).lower(): str(c) for c in columns}
    for cand in candidates:
        if cand in lookup:
            return lookup[cand]
    return None


def word_count(series: pd.Series) -> pd.Series:
    return series.str.split().str.len().fillna(0).astype(int)


# -----------------------------------------------------------------------------
# SPLIT BAZLI TEMİZLİK
# -----------------------------------------------------------------------------
def clean_split(
    df: pd.DataFrame,
    text_col: str,
    summary_col: str,
    split: str,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, Dict[str, int]]:
    """Bir split'e metin temizliği ve bozuk kayıt filtrelerini uygular."""
    stats: Dict[str, int] = {"split": split, "raw_rows": len(df)}

    # Temizlikten ÖNCE: hangi gürültü türünden kaç doküman etkilendi
    stats["docs_with_html"] = int(df[text_col].apply(has_pattern, pattern=HTML_RE).sum())
    stats["docs_with_url"] = int(df[text_col].apply(has_pattern, pattern=URL_RE).sum())
    stats["docs_with_email"] = int(df[text_col].apply(has_pattern, pattern=EMAIL_RE).sum())

    out = pd.DataFrame(
        {
            "clean_text": df[text_col].apply(clean_text),
            "clean_summary": df[summary_col].apply(clean_text),
        }
    )
    out["text_words"] = word_count(out["clean_text"])
    out["summary_words"] = word_count(out["clean_summary"])

    # 1) Boş kayıtlar: her split'te elenir
    keep = (out["clean_text"] != "") & (out["clean_summary"] != "")
    stats["dropped_empty"] = int((~keep).sum())
    out = out[keep]

    # 2) Kısa kayıtlar: train/validation'da (test'te yalnızca --filter-test ile)
    apply_short = split in ("train", "validation") or args.filter_test
    if apply_short:
        keep = out["text_words"] >= args.min_source_words
        stats["dropped_short_text"] = int((~keep).sum())
        out = out[keep]

        keep = out["summary_words"] >= args.min_summary_words
        stats["dropped_short_summary"] = int((~keep).sum())
        out = out[keep]
    else:
        stats["dropped_short_text"] = 0
        stats["dropped_short_summary"] = 0

    # 3) Aşırı uzun metinler: isteğe bağlı, yalnızca train
    if args.max_source_words and split == "train":
        keep = out["text_words"] <= args.max_source_words
        stats["dropped_too_long"] = int((~keep).sum())
        out = out[keep]
    else:
        stats["dropped_too_long"] = 0

    return out.reset_index(drop=True), stats


# -----------------------------------------------------------------------------
# ANA AKIŞ
# -----------------------------------------------------------------------------
def run_cleaning_pipeline(args: argparse.Namespace, dataset=None) -> Dict[str, pd.DataFrame]:
    print("=" * 78)
    print("   HAFTA 6: K1 TR-News VERİ TEMİZLEME BORU HATTI (resmi split'ler korunur)")
    print("=" * 78)

    if dataset is None:
        from datasets import load_dataset

        print("\n1. TR-News yükleniyor (batubayk/tr-news)...")
        dataset = load_dataset("batubayk/tr-news")

    missing = [s for s in SPLIT_OUTPUT_NAMES if s not in dataset]
    if missing:
        raise KeyError(f"Beklenen split'ler bulunamadı: {missing}. Mevcut: {list(dataset.keys())}")

    cleaned: Dict[str, pd.DataFrame] = {}
    all_stats: Dict[str, Dict[str, int]] = {}

    print("\n2. Her split ayrı ayrı temizleniyor...")
    for split in SPLIT_OUTPUT_NAMES:
        df = pd.DataFrame(dataset[split])
        text_col = find_column(df.columns, ["content", "text", "article"])
        summary_col = find_column(df.columns, ["abstract", "summary"])
        if text_col is None or summary_col is None:
            raise ValueError(f"Kaynak/özet sütunu bulunamadı. Sütunlar: {list(df.columns)}")

        cleaned[split], all_stats[split] = clean_split(df, text_col, summary_col, split, args)
        print(f" -> {split:<10}: {all_stats[split]['raw_rows']:,} -> {len(cleaned[split]):,} "
              f"(sütunlar: {text_col} / {summary_col})")

    print("\n3. Tekrar ve train/validation/test sızıntısı kontrol ediliyor...")
    train, val, test = cleaned["train"], cleaned["validation"], cleaned["test"]

    # Raporlama: validation/test içi tekrarlar (silinmez)
    all_stats["validation"]["duplicates_within_split_reported"] = int(val["clean_text"].duplicated().sum())
    all_stats["test"]["duplicates_within_split_reported"] = int(test["clean_text"].duplicated().sum())

    # validation'dan: test'te de bulunanlar
    test_texts = set(test["clean_text"])
    keep = ~val["clean_text"].isin(test_texts)
    all_stats["validation"]["dropped_overlap_with_test"] = int((~keep).sum())
    val = val[keep].reset_index(drop=True)

    # train'den: kendi içindeki tekrarlar
    keep = ~train["clean_text"].duplicated(keep="first")
    all_stats["train"]["dropped_duplicates_within_train"] = int((~keep).sum())
    train = train[keep]

    # train'den: validation/test ile örtüşenler
    heldout_texts = set(val["clean_text"]) | test_texts
    keep = ~train["clean_text"].isin(heldout_texts)
    all_stats["train"]["dropped_overlap_with_val_test"] = int((~keep).sum())
    train = train[keep].reset_index(drop=True)

    cleaned = {"train": train, "validation": val, "test": test}

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n4. Çıktılar kaydediliyor ({out_dir.resolve()})...")
    output_cols = ["clean_text", "clean_summary"]
    for split, file_name in SPLIT_OUTPUT_NAMES.items():
        path = out_dir / file_name
        cleaned[split][output_cols].to_csv(path, index=False, encoding="utf-8")
        all_stats[split]["final_rows"] = len(cleaned[split])
        print(f" -> {path} : {len(cleaned[split]):,} satır")

    stats_df = pd.DataFrame(all_stats.values()).fillna(0)
    for col in stats_df.columns:
        if col != "split":
            stats_df[col] = stats_df[col].astype(int)
    stats_path = out_dir / "cleaning_istatistikleri.csv"
    stats_df.to_csv(stats_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 78)
    print(" TEMİZLİK ÖZETİ (rapor için)")
    print("=" * 78)
    for _, row in stats_df.iterrows():
        raw, final = row["raw_rows"], row["final_rows"]
        dropped = raw - final
        print(f"\n[{row['split']}] ham: {raw:,} | son: {final:,} | elenen: {dropped:,} "
              f"(%{dropped / raw * 100:.2f})")
        print(f"   Gürültülü doküman (temizlik öncesi) -> HTML: {row['docs_with_html']:,} | "
              f"URL: {row['docs_with_url']:,} | e-posta: {row['docs_with_email']:,}")
        print(f"   Elenenler -> boş: {row['dropped_empty']:,} | kısa kaynak: {row['dropped_short_text']:,} | "
              f"kısa özet: {row['dropped_short_summary']:,} | aşırı uzun: {row['dropped_too_long']:,}")
        extras = {
            "train içi tekrar": row.get("dropped_duplicates_within_train", 0),
            "val/test ile örtüşme (train)": row.get("dropped_overlap_with_val_test", 0),
            "test ile örtüşme (val)": row.get("dropped_overlap_with_test", 0),
            "split içi tekrar (yalnızca rapor)": row.get("duplicates_within_split_reported", 0),
        }
        shown = {k: v for k, v in extras.items() if v}
        if shown:
            print("   Tekrar/sızıntı -> " + " | ".join(f"{k}: {v:,}" for k, v in shown.items()))
    print(f"\nİstatistik dosyası: {stats_path}")
    print("[BAŞARILI] Temizlik tamamlandı.")
    return cleaned


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TR-News temizleme (resmi split'ler korunur)")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--min-source-words", type=int, default=15)
    parser.add_argument("--min-summary-words", type=int, default=3)
    parser.add_argument(
        "--max-source-words", type=int, default=None,
        help="İsteğe bağlı: bundan uzun kaynak metinleri yalnızca train'den çıkarır (ör. 1200).",
    )
    parser.add_argument(
        "--filter-test", action="store_true",
        help="Kısa kayıt filtresini test setine de uygula (varsayılan: uygulanmaz).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    run_cleaning_pipeline(parse_args())
