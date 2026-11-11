"""
Rol: K4 - NLI Güvenlik Denetçisi
Görev: Özet cümlelerini kaynak metin pencereleriyle anlamsal olarak eşleştirmek.

Not: K1'in `chunk_text` fonksiyonu kelime sayısına göre böler ve cümleyi ortadan
kesebilir; NLI öncülü için uygun değil, bu yüzden burada cümle bazlı pencereleme var.
"""
from __future__ import annotations

import re
from typing import Dict, List, Set

from .text_utils import content_stems, tr_lower

# Küçük harfe çevrilmiş, noktalı kısaltmalar. Gerekirse genişlet.
_ABBREV = {
    "dr.", "prof.", "doç.", "yrd.", "av.", "sn.", "bkz.", "vb.", "vs.", "örn.",
    "yy.", "mah.", "cad.", "sok.", "no.", "tel.", "müh.", "uzm.", "gen.", "alb.",
    "çev.", "bk.", "t.c.", "a.ş.", "ltd.", "şti.", "mö.", "ms.", "st.", "yük.",
}
_INITIAL = re.compile(r"[A-ZÇĞİÖŞÜ]\.")  # "M. Kemal" gibi baş harfler
_BOUNDARY = re.compile(r'([.!?…]+["\'”’)\]]*)\s+')
_QUOTES = "\"'“”‘’()[]"


def split_sentences(text: str) -> List[str]:
    """Metni cümlelere böler.

    Bölmediği durumlar: kısaltmalar ("Dr."), baş harfler ("M."), sıra sayıları
    ("5. kat" - noktadan sonra küçük harf), ondalık/tarih ("3.5", "11.11.2026" -
    noktadan sonra boşluk olmadığı için zaten eşleşmez).
    """
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []

    sentences: List[str] = []
    start = 0
    for m in _BOUNDARY.finditer(text):
        punct = m.group(1)
        end = m.end(1)
        if punct.startswith("."):
            tokens = text[start:end].split()
            last = tokens[-1].strip(_QUOTES) if tokens else ""
            if tr_lower(last) in _ABBREV or _INITIAL.fullmatch(last):
                continue
            nxt = text[m.end():m.end() + 1]
            if nxt and nxt.islower():
                continue
        sentences.append(text[start:end].strip())
        start = m.end()

    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def build_windows(sentences: List[str], size: int, stride: int) -> List[str]:
    """Kaynak cümlelerinden `size` cümlelik kayan pencereler üretir (son pencere dahil)."""
    if not sentences:
        return []
    n = len(sentences)
    if n <= size:
        return [" ".join(sentences)]
    starts = list(range(0, n - size + 1, max(stride, 1)))
    if starts[-1] != n - size:
        starts.append(n - size)
    return [" ".join(sentences[i:i + size]) for i in starts]


def rank_windows(hyp: str, windows: List[str], stems: List[Set[str]], top_k: int) -> List[str]:
    """Hipotezle sözcük örtüşmesi en yüksek `top_k` pencereyi döndürür."""
    if len(windows) <= top_k:
        return list(windows)
    h = content_stems(hyp)
    if not h:
        return list(windows[:top_k])
    scores = [len(h & s) / len(h) for s in stems]
    order = sorted(range(len(windows)), key=lambda j: scores[j], reverse=True)
    return [windows[j] for j in order[:top_k]]


def split_and_match(
    premise: str,
    hypothesis: str,
    window_size: int = 3,
    stride: int = 1,
    top_k: int = 5,
) -> List[Dict]:
    """Özeti (hipotez) cümlelere, kaynağı (öncül) pencerelere böl; eşleştir.

    Dönüş: [{"index": i, "hypothesis": <özet cümlesi>, "premises": [<en alakalı pencereler>]}, ...]
    Kaynak ya da özet boşsa boş liste döner.
    """
    hyps = split_sentences(hypothesis)
    windows = build_windows(split_sentences(premise), window_size, stride)
    if not hyps or not windows:
        return []
    stems = [content_stems(w) for w in windows]
    return [
        {"index": i, "hypothesis": h, "premises": rank_windows(h, windows, stems, top_k)}
        for i, h in enumerate(hyps)
    ]
