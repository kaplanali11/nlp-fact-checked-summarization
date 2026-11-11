"""Türkçeye duyarlı küçük metin yardımcıları."""
from __future__ import annotations

import re
from typing import Set

# Python'da "İ".lower() -> "i̇" (birleştirici noktalı) ve "I".lower() -> "i" olur;
# Türkçede doğrusu "i" ve "ı". Önce elle eşliyoruz.
_TR_MAP = str.maketrans({"İ": "i", "I": "ı"})
# Sadece sözcük örtüşmesi (retrieval) için: aksanları sil. K3'ün inference.py
# öz-testi gibi ASCII'ye çevrilmiş metinler ("Universitesi", "arastirma") ile
# aksanlı metinler aynı kök olarak eşleşsin.
_FOLD = str.maketrans("çğışöüâîû", "cgisouaiu")
_WORD = re.compile(r"\w+", re.UNICODE)
_NUM = re.compile(r"\d+(?:[.,]\d+)*")
# T5/mT5 çıktısında sızabilen sentinel / özel token artıkları
_SPECIAL = re.compile(r"<extra_id_\d+>|<pad>|</s>|<unk>")


def tr_lower(text: str) -> str:
    return text.translate(_TR_MAP).lower()


def fold_diacritics(text: str) -> str:
    return tr_lower(text).translate(_FOLD)


def content_stems(text: str, stem_len: int = 5, min_len: int = 3) -> Set[str]:
    """Kaba 'stemmer': Türkçe sondan eklemeli olduğu için ilk `stem_len` karakteri al.

    Sadece pencere sıralamasında (retrieval) kullanılır, NLI kararını etkilemez.
    Kısa/ek benzeri token'lar (<min_len) atılır; aksanlar silinir.
    """
    return {t[:stem_len] for t in _WORD.findall(fold_diacritics(text)) if len(t) >= min_len}


def extract_numbers(text: str) -> Set[str]:
    """Metindeki sayıları ayraçlardan arındırılmış rakam dizisi olarak döndürür.

    "1.500" ve "1,500" -> "1500". Ondalıklarda ("3,5" -> "35") çakışma olabilir;
    bu kasıtlı olarak toleranslı (false-positive yerine false-negative tercih edildi).
    """
    return {re.sub(r"[.,]", "", m) for m in _NUM.findall(text)}


def clean_summary(text: str) -> str:
    """K3'ün ürettiği özetten sentinel token artıklarını temizler, boşlukları toplar."""
    return re.sub(r"\s+", " ", _SPECIAL.sub(" ", text or "")).strip()
