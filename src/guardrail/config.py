"""K4 (NLI Güvenlik Denetçisi) yapılandırması.

Tüm eşikler ve model ayarları tek yerde toplanır. Eşikleri körlemesine
varsayılan bırakma: validation setinde (K1'in verdiği) kalibre etmen gerekir.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class GuardrailConfig:
    # --- Model ---------------------------------------------------------
    # DİKKAT: "dbmdz/bert-base-turkish-cased" (eski iskeletteki değer) NLI için
    # İNCE AYARLI DEĞİL; sınıflandırma kafası rastgele başlar ve çıktısı anlamsızdır.
    # NLI (entailment/neutral/contradiction) için ince ayarlı bir model gerekir.
    # Adı HF Hub'da doğrula; erişemiyorsan buradan değiştir.
    model_name: str = "emrecan/convbert-base-turkish-mc4-cased-allnli_tr"
    # Modelin config.id2label'ı "LABEL_0" gibi anlamsızsa, indeks sırasına göre
    # etiket isimlerini ver. Örn: ("entailment", "neutral", "contradiction")
    label_order: Optional[Sequence[str]] = None
    device: Optional[str] = None  # None -> cuda varsa cuda, yoksa cpu
    max_length: int = 512
    batch_size: int = 16

    # --- Öncül (kaynak metin) pencereleme ------------------------------
    # Kaynak metin 512 token'a sığmaz. Cümle pencerelerine bölüp her özet
    # cümlesini en alakalı top_k pencereyle eşleştiriyoruz.
    window_size: int = 3
    window_stride: int = 1
    top_k_windows: int = 5

    # --- Karar eşikleri -------------------------------------------------
    entailment_threshold: float = 0.60
    contradiction_threshold: float = 0.50

    # --- Ek kural: sayı koruması ---------------------------------------
    # NLI modelleri "500" -> "900" gibi sayı halüsinasyonlarını sık kaçırır.
    # Özetteki bir sayı kaynakta hiç geçmiyorsa entailment'ı düşürür.
    check_numbers: bool = True
