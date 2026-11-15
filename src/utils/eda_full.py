"""
HAFTA 6 - K1: TÜRKÇE HABER VERİ KÜMELERİ TAM VERİ EDA BETİĞİ

Veri kümeleri:
  1) TR-News: batubayk/tr-news
  2) MLSUM Türkçe: reciTAL/mlsum (config: tu)

Bu sürüm örneklem kullanmaz. Her veri kümesinin mevcut tüm split'lerini
(train / validation / test vb.) ayrı ayrı ve eksiksiz işler.

Bellek yönetimi:
  - Hugging Face Dataset kayıtları batch_size kadar okunur.
  - Tokenizer, tokenizer_batch_size kadar metni birlikte işler.
  - Her split tamamlanınca özet sonuçlar CSV dosyalarına eklenmek üzere bellekte
    tutulur; ham metinlerin tamamı pandas DataFrame'e dönüştürülmez.

Kurulum:
  pip install numpy pandas nltk datasets transformers sentencepiece

Çalıştırma:
  python eda_full.py

İsteğe bağlı:
  python eda_full.py --batch-size 128 --tokenizer-batch-size 8
  python eda_full.py --skip-tokenizer
  python eda_full.py --output-dir eda_sonuclari

Notlar:
  - Varsayılan çalışma tüm kayıtları, tüm split'lerde işler.
  - --skip-tokenizer seçilirse gerçek subword ölçümü yapılmaz; token kapsama
    değerleri yalnızca yaklaşık 1.35 katsayısıyla raporlanır.
  - Novel n-gram oranı sözcüksel farklılık göstergesidir; tek başına halüsinasyon
    kanıtı değildir. Olgusal tutarlılık için NLI değerlendirmesi gerekir.
"""

from __future__ import annotations

import argparse
import gc
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import nltk
import numpy as np
import pandas as pd
from datasets import DatasetDict, load_dataset
from nltk.corpus import stopwords
from nltk.util import ngrams
from transformers import AutoTokenizer


# -----------------------------------------------------------------------------
# AYARLAR
# -----------------------------------------------------------------------------
MODEL_NAME = "savasy/mt5-mlsum-turkish-summarization"
PERCENTILES = (50, 90, 95, 98, 99)
DEFAULT_BATCH_SIZE = 256
DEFAULT_TOKENIZER_BATCH_SIZE = 16

DATASET_CONFIGS = [
    {
        "name": "TR-News",
        "repo_id": "batubayk/tr-news",
        "config": None,
    },
    {
        "name": "MLSUM Türkçe",
        "repo_id": "reciTAL/mlsum",
        "config": "tu",
    },
]

# Kesme işaretini korur. Böylece "Türkiye'nin" ifadesi "türkiye" + "nin"
# biçiminde iki ayrı sözcüğe bölünmez.
WORD_PATTERN = re.compile(
    r"[a-zA-Zçğıöşüâîû]+(?:['’][a-zA-Zçğıöşüâîû]+)*",
    flags=re.IGNORECASE,
)
HTML_PATTERN = re.compile(r"<[^>]+>")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)
TURKISH_CHAR_PATTERN = re.compile(r"[çğıöşüÇĞİÖŞÜ]")

# Haber dilinde sık geçen, içerik analizinde ayrı tutulması tercih edilen kelimeler.
DOMAIN_STOPWORDS = {
    "dedi", "açıkladı", "belirtti", "sonra", "göre", "üzerine",
    "yaptı", "eden", "oldu", "bir",
}

FALLBACK_TR_STOPWORDS = {
    "acaba", "ama", "aslında", "az", "bazı", "belki", "biri", "birkaç",
    "birşey", "biz", "bu", "çok", "çünkü", "da", "daha", "de", "defa",
    "diye", "en", "gibi", "hem", "hep", "hepsi", "her", "hiç", "ile",
    "ise", "için", "kez", "ki", "kim", "mı", "mi", "mu", "mü", "nasıl",
    "ne", "neden", "nerde", "nerede", "nereye", "niçin", "niye", "o", "sanki",
    "sen", "siz", "şey", "şu", "tüm", "ve", "veya", "ya", "yani",
    "olarak", "olan", "olduğu", "olup", "var", "yok",
}


# -----------------------------------------------------------------------------
# GENEL YARDIMCILAR
# -----------------------------------------------------------------------------
def tr_lower(text: object) -> str:
    """Türkçedeki I/İ harflerini dikkate alarak küçük harfe dönüştürür."""
    if not isinstance(text, str):
        return ""
    return text.replace("İ", "i").replace("I", "ı").lower()


def tokenize_clean(text: object) -> List[str]:
    """Noktalama işaretlerini ayırır; Türkçe harfleri ve kesme işaretini korur."""
    return WORD_PATTERN.findall(tr_lower(text))


def is_missing(value: Any) -> bool:
    """None, NaN ve pandas NA değerlerini güvenli biçimde saptar."""
    if value is None:
        return True
    try:
        result = pd.isna(value)
        return bool(result) if np.isscalar(result) else False
    except (TypeError, ValueError):
        return False


def clean_text(value: Any) -> str:
    """Eksik değeri 'nan' metnine çevirmeden boş metin olarak normalize eder."""
    if is_missing(value):
        return ""
    return str(value).strip()


def prepare_nltk_resources() -> set:
    """NLTK kaynaklarını indirir/yükler; erişilemezse sınırlı yedek listeyi kullanır."""
    for resource in ("punkt", "punkt_tab", "stopwords"):
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass

    try:
        words = {tr_lower(word) for word in stopwords.words("turkish")}
        print("[BİLGİ] NLTK Türkçe stopword listesi yüklendi.")
    except LookupError:
        words = {tr_lower(word) for word in FALLBACK_TR_STOPWORDS}
        print(
            "[UYARI] NLTK Türkçe stopword listesi bulunamadı. "
            "Sınırlı yedek liste kullanılacak."
        )

    return words | {tr_lower(word) for word in DOMAIN_STOPWORDS}


def sentence_count(text: str) -> int:
    """Türkçe NLTK cümle ayırıcısını kullanır; kaynak yoksa regex'e döner."""
    if not text:
        return 0
    try:
        sentences = nltk.sent_tokenize(text, language="turkish")
        return sum(1 for sentence in sentences if sentence.strip())
    except (LookupError, ValueError):
        pieces = re.split(r"(?<=[.!?…])\s+", text.strip())
        return sum(1 for piece in pieces if piece.strip())


def find_column(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    """Sütun adını büyük/küçük harf duyarsız biçimde bulur."""
    lookup = {str(column).lower(): str(column) for column in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def identify_columns(split_data: Any) -> Tuple[str, str]:
    """Kaynak metin ve özet sütunlarını veri kümesinin özelliklerinden bulur."""
    columns = list(split_data.column_names)
    text_col = find_column(
        columns, ["content", "text", "article", "document", "body"]
    )
    summary_col = find_column(
        columns, ["abstract", "summary", "target", "highlights"]
    )
    if text_col is None or summary_col is None:
        raise ValueError(
            "Kaynak metin/özet sütunu bulunamadı. "
            f"Mevcut sütunlar: {columns}"
        )
    return text_col, summary_col


def load_hf_dataset(repo_id: str, config: Optional[str] = None):
    """Hugging Face veri kümesini yükler."""
    if config is None:
        return load_dataset(repo_id)
    # MLSUM yapılandırması bazı datasets sürümlerinde trust_remote_code
    # parametresini kabul etmeyebilir; iki biçim de denenir.
    try:
        return load_dataset(repo_id, config, trust_remote_code=True)
    except TypeError:
        return load_dataset(repo_id, config)


def load_tokenizer():
    """Gerçek subword uzunluklarını ölçmek için tokenizer yükler."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
        print(f"[BAŞARILI] Fast tokenizer yüklendi: {MODEL_NAME}")
        return tokenizer
    except Exception as fast_error:
        print(f"[UYARI] Fast tokenizer yüklenemedi: {fast_error}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
            print(f"[BAŞARILI] Slow tokenizer yüklendi: {MODEL_NAME}")
            return tokenizer
        except Exception as slow_error:
            print(f"[UYARI] Tokenizer yüklenemedi: {slow_error}")
            print(
                "[UYARI] Tüm veriler yine analiz edilecek; token uzunluğu ve "
                "kapsama için 1.35 katsayılı yaklaşık tahmin kullanılacak."
            )
            return None


def get_token_lengths(
    texts: Sequence[str], tokenizer: Any, tokenizer_batch_size: int
) -> List[int]:
    """Kesme yapmadan ve özel tokenları dahil ederek gerçek token uzunluklarını ölçer."""
    lengths: List[int] = []
    for start in range(0, len(texts), tokenizer_batch_size):
        batch = list(texts[start : start + tokenizer_batch_size])
        encoded = tokenizer(
            batch,
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )
        lengths.extend(len(ids) for ids in encoded["input_ids"])
    return lengths


def percentile_values(values: Sequence[int], percentiles=PERCENTILES) -> List[int]:
    if not values:
        return [0 for _ in percentiles]
    return [int(round(value)) for value in np.percentile(values, percentiles)]


def get_novel_ngram_ratio(
    source_tokens: Sequence[str], summary_tokens: Sequence[str], n: int
) -> float:
    """Özet n-gramlarının kaynakta aynen bulunmayan kısmını hesaplar."""
    if len(source_tokens) < n or len(summary_tokens) < n:
        return 0.0
    source_ngrams = set(ngrams(source_tokens, n))
    summary_ngrams = set(ngrams(summary_tokens, n))
    if not summary_ngrams:
        return 0.0
    return len(summary_ngrams - source_ngrams) / len(summary_ngrams)


# -----------------------------------------------------------------------------
# SPLIT BAZLI BİRİKTİRİCİ
# -----------------------------------------------------------------------------
class SplitAccumulator:
    """Bir split'in tüm kayıtlarına ait istatistikleri artımlı olarak biriktirir."""

    def __init__(self, total_rows: int):
        self.total_rows = total_rows
        self.processed_rows = 0
        self.missing_text = 0
        self.missing_summary = 0
        self.empty_text = 0
        self.empty_summary = 0
        self.dropped_rows = 0
        self.valid_docs = 0

        self.text_word_lengths: List[int] = []
        self.summary_word_lengths: List[int] = []
        self.text_sentence_lengths: List[int] = []
        self.summary_sentence_lengths: List[int] = []
        self.text_token_lengths: List[int] = []
        self.summary_token_lengths: List[int] = []

        self.total_source_words = 0
        self.total_source_tokens = 0
        self.stopword_count = 0
        self.corpus_vocab = set()
        self.document_ttr_sum = 0.0
        self.document_ttr_count = 0
        self.word_counter = Counter()
        self.bigram_counter = Counter()

        self.html_count = 0
        self.url_count = 0
        self.email_count = 0
        self.turkish_char_count = 0

        self.novelty_sums = {1: 0.0, 2: 0.0, 3: 0.0}
        self.novelty_docs = {1: 0, 2: 0, 3: 0}

    def add_batch(
        self,
        raw_texts: Sequence[Any],
        raw_summaries: Sequence[Any],
        tokenizer: Any,
        stopwords_set: set,
        tokenizer_batch_size: int,
    ) -> None:
        """Bir batch'i temizler, ölçer ve split istatistiklerine ekler."""
        valid_texts: List[str] = []
        valid_summaries: List[str] = []

        for raw_text, raw_summary in zip(raw_texts, raw_summaries):
            self.processed_rows += 1

            text_missing = is_missing(raw_text)
            summary_missing = is_missing(raw_summary)
            if text_missing:
                self.missing_text += 1
            if summary_missing:
                self.missing_summary += 1

            text = clean_text(raw_text)
            summary = clean_text(raw_summary)

            if not text and not text_missing:
                self.empty_text += 1
            if not summary and not summary_missing:
                self.empty_summary += 1

            if not text or not summary:
                self.dropped_rows += 1
                continue

            valid_texts.append(text)
            valid_summaries.append(summary)

        if not valid_texts:
            return

        text_word_lengths = [len(text.split()) for text in valid_texts]
        summary_word_lengths = [len(summary.split()) for summary in valid_summaries]
        text_sentence_lengths = [sentence_count(text) for text in valid_texts]
        summary_sentence_lengths = [sentence_count(summary) for summary in valid_summaries]

        self.text_word_lengths.extend(text_word_lengths)
        self.summary_word_lengths.extend(summary_word_lengths)
        self.text_sentence_lengths.extend(text_sentence_lengths)
        self.summary_sentence_lengths.extend(summary_sentence_lengths)

        if tokenizer is not None:
            self.text_token_lengths.extend(
                get_token_lengths(valid_texts, tokenizer, tokenizer_batch_size)
            )
            self.summary_token_lengths.extend(
                get_token_lengths(valid_summaries, tokenizer, tokenizer_batch_size)
            )

        self.valid_docs += len(valid_texts)

        for text, summary, text_word_count in zip(
            valid_texts, valid_summaries, text_word_lengths
        ):
            source_tokens = tokenize_clean(text)
            summary_tokens = tokenize_clean(summary)
            source_tokens_no_stop = [
                token for token in source_tokens if token not in stopwords_set
            ]

            self.total_source_words += text_word_count
            self.total_source_tokens += len(source_tokens)
            self.stopword_count += len(source_tokens) - len(source_tokens_no_stop)
            self.corpus_vocab.update(source_tokens)

            if source_tokens:
                self.document_ttr_sum += len(set(source_tokens)) / len(source_tokens)
                self.document_ttr_count += 1

            self.word_counter.update(source_tokens_no_stop)
            if len(source_tokens_no_stop) >= 2:
                self.bigram_counter.update(ngrams(source_tokens_no_stop, 2))

            self.html_count += int(bool(HTML_PATTERN.search(text)))
            self.url_count += int(bool(URL_PATTERN.search(text)))
            self.email_count += int(bool(EMAIL_PATTERN.search(text)))
            self.turkish_char_count += int(bool(TURKISH_CHAR_PATTERN.search(text)))

            for n in (1, 2, 3):
                self.novelty_sums[n] += get_novel_ngram_ratio(
                    source_tokens, summary_tokens, n
                )
                self.novelty_docs[n] += 1


def summarize_split(
    dataset_name: str,
    split_name: str,
    accumulator: SplitAccumulator,
    tokenizer: Any,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split sonuçlarını ekrana basar ve CSV için satırlar üretir."""
    acc = accumulator
    valid = acc.valid_docs

    print("\n" + "-" * 88)
    print(f"SPLIT SONUÇLARI: {dataset_name} / {split_name}")
    print("-" * 88)
    print(f"Split'teki toplam kayıt                 : {acc.total_rows:,}")
    print(f"İşlenen kayıt                           : {acc.processed_rows:,}")
    print(f"Eksik kaynak metin                      : {acc.missing_text:,}")
    print(f"Eksik özet                              : {acc.missing_summary:,}")
    print(f"Boş kaynak metin                        : {acc.empty_text:,}")
    print(f"Boş özet                                : {acc.empty_summary:,}")
    print(f"Geçersiz olduğu için çıkarılan kayıt    : {acc.dropped_rows:,}")
    print(f"Analize dahil edilen geçerli kayıt      : {valid:,}")

    if valid == 0:
        print("[UYARI] Bu split'te analiz edilebilir kayıt bulunamadı.")
        return (
            {
                "dataset": dataset_name,
                "split": split_name,
                "total_rows": acc.total_rows,
                "valid_docs": 0,
                "dropped_rows": acc.dropped_rows,
            },
            [],
            [],
        )

    avg_text_words = float(np.mean(acc.text_word_lengths))
    avg_summary_words = float(np.mean(acc.summary_word_lengths))
    avg_text_sentences = float(np.mean(acc.text_sentence_lengths))
    avg_summary_sentences = float(np.mean(acc.summary_sentence_lengths))

    print("\n1. TEMEL METİN VE CÜMLE İSTATİSTİKLERİ")
    print(f"Ortalama kaynak kelime sayısı           : {avg_text_words:.2f}")
    print(f"Ortalama özet kelime sayısı             : {avg_summary_words:.2f}")
    print(f"Ortalama kaynak cümle sayısı            : {avg_text_sentences:.2f}")
    print(f"Ortalama özet cümle sayısı              : {avg_summary_sentences:.2f}")

    print("\n2. TOKENIZER / BAĞLAM PENCERESİ KAPSAMA ANALİZİ")
    text_word_p = percentile_values(acc.text_word_lengths)
    summary_word_p = percentile_values(acc.summary_word_lengths)
    if tokenizer is not None and acc.text_token_lengths:
        text_token_p = percentile_values(acc.text_token_lengths)
        summary_token_p = percentile_values(acc.summary_token_lengths)
        print("Yüzdelik  Kaynak Kelime  Özet Kelime  Kaynak Subword  Özet Subword")
        for i, percentile in enumerate(PERCENTILES):
            print(
                f"%{percentile:<7} {text_word_p[i]:>13,} {summary_word_p[i]:>12,} "
                f"{text_token_p[i]:>15,} {summary_token_p[i]:>13,}"
            )
        input_lengths = np.asarray(acc.text_token_lengths, dtype=np.int32)
        target_lengths = np.asarray(acc.summary_token_lengths, dtype=np.int32)
        measurement = "gerçek tokenizer ölçümü (tüm geçerli kayıtlar)"
        expansion = (
            sum(acc.text_token_lengths) / acc.total_source_words
            if acc.total_source_words
            else 0.0
        )
        print(f"Ağırlıklı subword genişleme çarpanı     : {expansion:.3f} token/kelime")
    else:
        input_lengths = np.asarray(acc.text_word_lengths, dtype=np.float64) * 1.35
        target_lengths = np.asarray(acc.summary_word_lengths, dtype=np.float64) * 1.35
        measurement = "1.35 katsayılı yaklaşık tahmin (tüm geçerli kayıtlar)"
        print("Yüzdelik  Kaynak Kelime  Özet Kelime  Kaynak Tahmini Token  Özet Tahmini Token")
        for i, percentile in enumerate(PERCENTILES):
            print(
                f"%{percentile:<7} {text_word_p[i]:>13,} {summary_word_p[i]:>12,} "
                f"{int(round(text_word_p[i] * 1.35)):>21,} "
                f"{int(round(summary_word_p[i] * 1.35)):>19,}"
            )

    cov_512 = float(np.mean(input_lengths <= 512) * 100)
    cov_1024 = float(np.mean(input_lengths <= 1024) * 100)
    cov_target_128 = float(np.mean(target_lengths <= 128) * 100)
    print(f"Ölçüm türü                              : {measurement}")
    print(f"max_input_length=512 kapsama            : %{cov_512:.2f}")
    print(f"max_input_length=1024 kapsama           : %{cov_1024:.2f}")
    print(f"max_target_length=128 kapsama           : %{cov_target_128:.2f}")

    print("\n3. KELİME FREKANSI, STOPWORDS VE TTR ANALİZİ")
    stopword_ratio = (
        acc.stopword_count / acc.total_source_tokens * 100
        if acc.total_source_tokens
        else 0.0
    )
    corpus_ttr = (
        len(acc.corpus_vocab) / acc.total_source_tokens
        if acc.total_source_tokens
        else 0.0
    )
    mean_document_ttr = (
        acc.document_ttr_sum / acc.document_ttr_count
        if acc.document_ttr_count
        else 0.0
    )
    print(f"Stopword oranı (kaynak metin)            : %{stopword_ratio:.2f}")
    print(f"Benzersiz kelime sayısı                  : {len(acc.corpus_vocab):,}")
    print(f"Korpus TTR                               : {corpus_ttr:.4f}")
    print(f"Belge bazlı ortalama TTR                 : {mean_document_ttr:.4f}")

    print("\nEn sık geçen 10 kelime (stopwords hariç):")
    word_rows: List[Dict[str, Any]] = []
    for rank, (word, frequency) in enumerate(acc.word_counter.most_common(10), start=1):
        print(f"  {rank:>2}. {word}: {frequency:,}")
        word_rows.append(
            {
                "dataset": dataset_name,
                "split": split_name,
                "rank": rank,
                "word": word,
                "frequency": frequency,
            }
        )

    print("\nEn sık geçen 5 bigram (belge sınırları korunarak):")
    bigram_rows: List[Dict[str, Any]] = []
    for rank, (bigram, frequency) in enumerate(acc.bigram_counter.most_common(5), start=1):
        phrase = " ".join(bigram)
        print(f"  {rank:>2}. {phrase}: {frequency:,}")
        bigram_rows.append(
            {
                "dataset": dataset_name,
                "split": split_name,
                "rank": rank,
                "bigram": phrase,
                "frequency": frequency,
            }
        )

    print("\n4. GÜRÜLTÜ VE İÇERİK BOZULMA ANALİZİ")
    turkish_char_missing = valid - acc.turkish_char_count
    print(
        f"HTML etiketi içeren metin                : {acc.html_count:,} "
        f"(%{acc.html_count / valid * 100:.2f})"
    )
    print(
        f"URL/link içeren metin                    : {acc.url_count:,} "
        f"(%{acc.url_count / valid * 100:.2f})"
    )
    print(
        f"E-posta içeren metin                     : {acc.email_count:,} "
        f"(%{acc.email_count / valid * 100:.2f})"
    )
    print(
        f"Türkçe karakter içermeyen metin          : {turkish_char_missing:,} "
        f"(%{turkish_char_missing / valid * 100:.2f})"
    )

    print("\n5. ÖZET SOYUTLAMA (NOVELTY) / SÖZCÜKSEL FARKLILIK ANALİZİ")
    novelty_results = {}
    for n in (1, 2, 3):
        average = (
            acc.novelty_sums[n] / acc.novelty_docs[n] * 100
            if acc.novelty_docs[n]
            else 0.0
        )
        novelty_results[n] = average
        print(f"Kaynakta bulunmayan yeni {n}-gram oranı : %{average:.2f}")

    print("\nYorum:")
    print(
        "  Novel n-gram oranı, özetin kaynak metindeki kelime dizilerinden "
        "ne kadar farklılaştığını gösteren sözcüksel bir ölçüdür."
    )
    print(
        "  Yüksek novelty tek başına halüsinasyon kanıtı değildir; doğru bir "
        "yeniden ifade de kaynakta aynen geçmeyen n-gramlar üretebilir."
    )
    print("  Olgusal tutarlılık için K4'ün NLI modeli ve anlamsal inceleme kullanılmalıdır.")

    summary_row: Dict[str, Any] = {
        "dataset": dataset_name,
        "split": split_name,
        "total_rows": acc.total_rows,
        "processed_rows": acc.processed_rows,
        "valid_docs": valid,
        "missing_text": acc.missing_text,
        "missing_summary": acc.missing_summary,
        "empty_text": acc.empty_text,
        "empty_summary": acc.empty_summary,
        "dropped_rows": acc.dropped_rows,
        "avg_source_words": avg_text_words,
        "avg_summary_words": avg_summary_words,
        "avg_source_sentences": avg_text_sentences,
        "avg_summary_sentences": avg_summary_sentences,
        "stopword_ratio_pct": stopword_ratio,
        "vocabulary_size": len(acc.corpus_vocab),
        "corpus_ttr": corpus_ttr,
        "mean_document_ttr": mean_document_ttr,
        "measurement_type": measurement,
        "coverage_input_512_pct": cov_512,
        "coverage_input_1024_pct": cov_1024,
        "coverage_target_128_pct": cov_target_128,
        "html_count": acc.html_count,
        "url_count": acc.url_count,
        "email_count": acc.email_count,
        "no_turkish_char_count": turkish_char_missing,
        "novel_1gram_pct": novelty_results[1],
        "novel_2gram_pct": novelty_results[2],
        "novel_3gram_pct": novelty_results[3],
    }
    for i, percentile in enumerate(PERCENTILES):
        summary_row[f"source_words_p{percentile}"] = text_word_p[i]
        summary_row[f"summary_words_p{percentile}"] = summary_word_p[i]
        if tokenizer is not None and acc.text_token_lengths:
            summary_row[f"source_subwords_p{percentile}"] = percentile_values(
                acc.text_token_lengths
            )[i]
            summary_row[f"summary_subwords_p{percentile}"] = percentile_values(
                acc.summary_token_lengths
            )[i]

    print(f"\n[TAMAMLANDI] {dataset_name} / {split_name}: {valid:,} kayıt analiz edildi.")
    return summary_row, word_rows, bigram_rows


def process_split(
    dataset_name: str,
    split_name: str,
    split_data: Any,
    batch_size: int,
    tokenizer: Any,
    tokenizer_batch_size: int,
    stopwords_set: set,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Bir split'in tamamını batch'ler halinde işler."""
    total_rows = len(split_data)
    text_col, summary_col = identify_columns(split_data)
    print(f"\nKaynak metin sütunu: {text_col} | Özet sütunu: {summary_col}")
    print(f"Tam split işlenecek: {total_rows:,} kayıt | Batch: {batch_size}")

    accumulator = SplitAccumulator(total_rows)
    for batch_number, batch in enumerate(split_data.iter(batch_size=batch_size), start=1):
        accumulator.add_batch(
            batch[text_col],
            batch[summary_col],
            tokenizer,
            stopwords_set,
            tokenizer_batch_size,
        )

        if accumulator.processed_rows % 10000 < batch_size or accumulator.processed_rows == total_rows:
            percentage = (
                accumulator.processed_rows / total_rows * 100 if total_rows else 100.0
            )
            print(
                f"  İlerleme: {accumulator.processed_rows:,}/{total_rows:,} "
                f"(%{percentage:.1f})",
                flush=True,
            )

    return summarize_split(dataset_name, split_name, accumulator, tokenizer)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="K1 Hafta 6: TR-News ve MLSUM Türkçe için tam veri EDA"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Dataset okuma batch boyutu (varsayılan: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--tokenizer-batch-size",
        type=int,
        default=DEFAULT_TOKENIZER_BATCH_SIZE,
        help=(
            "Tokenizer'a bir seferde verilecek metin sayısı "
            f"(varsayılan: {DEFAULT_TOKENIZER_BATCH_SIZE})"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="eda_sonuclari",
        help="CSV sonuçlarının kaydedileceği klasör (varsayılan: eda_sonuclari)",
    )
    parser.add_argument(
        "--skip-tokenizer",
        action="store_true",
        help="Gerçek tokenizer ölçümünü atla; yaklaşık 1.35 katsayısı kullan.",
    )
    args = parser.parse_args()

    if args.batch_size <= 0 or args.tokenizer_batch_size <= 0:
        parser.error("Batch boyutları pozitif tam sayı olmalıdır.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("HAFTA 6: K1 TÜRKÇE HABER VERİ KÜMELERİ TAM VERİ EDA BETİĞİ")
    print("=" * 88)
    print("Örneklem: YOK | Her split'teki TÜM kayıtlar işlenecek.")
    print(f"Dataset batch boyutu: {args.batch_size}")
    print(f"Tokenizer batch boyutu: {args.tokenizer_batch_size}")
    print(f"Çıktı klasörü: {output_dir.resolve()}")

    stopwords_set = prepare_nltk_resources()
    tokenizer = None if args.skip_tokenizer else load_tokenizer()

    all_summary_rows: List[Dict[str, Any]] = []
    all_word_rows: List[Dict[str, Any]] = []
    all_bigram_rows: List[Dict[str, Any]] = []
    failures: List[str] = []
    successful_splits = 0
    total_splits = 0

    for dataset_config in DATASET_CONFIGS:
        dataset_name = dataset_config["name"]
        repo_id = dataset_config["repo_id"]
        config = dataset_config["config"]

        print("\n" + "=" * 88)
        print(f"VERİ KÜMESİ: {dataset_name}")
        print("=" * 88)
        print(f"[1/3] Yükleniyor: {repo_id}" + (f" ({config})" if config else ""))

        try:
            dataset = load_hf_dataset(repo_id, config)
        except Exception as exc:
            message = f"{dataset_name} yüklenemedi: {exc}"
            failures.append(message)
            print(f"[HATA] {message}", file=sys.stderr)
            continue

        if isinstance(dataset, DatasetDict):
            split_items = list(dataset.items())
        else:
            split_items = [("data", dataset)]

        print("Bulunan split'ler:")
        for split_name, split_data in split_items:
            print(f"  - {split_name}: {len(split_data):,} kayıt")

        for split_name, split_data in split_items:
            total_splits += 1
            print("\n" + "-" * 88)
            print(f"[2/3] TAM VERİ ANALİZİ: {dataset_name} / {split_name}")
            print("-" * 88)
            try:
                summary_row, word_rows, bigram_rows = process_split(
                    dataset_name=dataset_name,
                    split_name=split_name,
                    split_data=split_data,
                    batch_size=args.batch_size,
                    tokenizer=tokenizer,
                    tokenizer_batch_size=args.tokenizer_batch_size,
                    stopwords_set=stopwords_set,
                )
                all_summary_rows.append(summary_row)
                all_word_rows.extend(word_rows)
                all_bigram_rows.extend(bigram_rows)
                successful_splits += 1
            except Exception as exc:
                message = f"{dataset_name}/{split_name} analizi başarısız: {exc}"
                failures.append(message)
                print(f"[HATA] {message}", file=sys.stderr)

        # Büyük dataset nesnesini bir sonraki veri kümesini yüklemeden bırak.
        del dataset
        gc.collect()

    print("\n" + "=" * 88)
    print("[3/3] SONUÇLAR CSV DOSYALARINA KAYDEDİLİYOR")
    print("=" * 88)

    if all_summary_rows:
        summary_path = output_dir / "eda_ozet_istatistikleri.csv"
        pd.DataFrame(all_summary_rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"Özet istatistikler : {summary_path.resolve()}")

    if all_word_rows:
        words_path = output_dir / "en_sik_kelimeler.csv"
        pd.DataFrame(all_word_rows).to_csv(words_path, index=False, encoding="utf-8-sig")
        print(f"En sık kelimeler   : {words_path.resolve()}")

    if all_bigram_rows:
        bigrams_path = output_dir / "en_sik_bigramlar.csv"
        pd.DataFrame(all_bigram_rows).to_csv(bigrams_path, index=False, encoding="utf-8-sig")
        print(f"En sık bigramlar   : {bigrams_path.resolve()}")

    print("\n" + "=" * 88)
    print("GENEL SONUÇ")
    print("=" * 88)
    print(f"Başarıyla analiz edilen split: {successful_splits}/{total_splits}")
    for failure in failures:
        print(f"- {failure}")

    if successful_splits == 0:
        print("Hiçbir split analiz edilemedi. Kurulum, internet ve veri erişimini kontrol edin.")
        return 1
    if failures:
        print("Bazı split'ler analiz edilemedi. Hata mesajlarını kontrol edin.")
        return 1

    print("TR-News ve MLSUM Türkçe veri kümelerinin bulunan tüm split'leri analiz edildi.")
    print("CSV raporları çıktı klasörüne kaydedildi.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
