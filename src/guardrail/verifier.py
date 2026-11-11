"""K4 karar motoru: özet cümlelerini kaynak metne karşı denetler.

Akış (her özet cümlesi = hipotez):
  1. split_and_match: kaynak cümlelere/pencerelere, özet cümlelere bölünür;
     her cümle için sözcük örtüşmesine göre en iyi top_k pencere seçilir.
  2. Tüm (pencere, hipotez) çiftleri tek seferde NLI modeline verilir.
  3. Pencereler arası: entailment için MAX, çelişki için MAX alınır.
  4. Karar: entailment >= eşik -> ENTAILMENT; aksi halde contradiction >= eşik
     ve entailment'tan büyükse -> CONTRADICTION; değilse -> NEUTRAL (dayanaksız).
  5. Sayı koruması: özetteki sayı kaynakta yoksa ENTAILMENT, NEUTRAL'e düşürülür.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from .berturk_nli import NLIBackend, Probs
from .config import GuardrailConfig
from .schemas import SentenceResult, Verdict, VerificationReport
from .sentence_matcher import split_and_match
from .text_utils import clean_summary, extract_numbers


class GuardrailVerifier:
    def __init__(
        self,
        config: Optional[GuardrailConfig] = None,
        backend: Optional[NLIBackend] = None,
    ) -> None:
        self.config = config or GuardrailConfig()
        if backend is None:
            from .berturk_nli import FactCheckNLI  # ağır import'u gerçekten gerekince yap

            backend = FactCheckNLI(
                self.config.model_name,
                device=self.config.device,
                max_length=self.config.max_length,
                batch_size=self.config.batch_size,
                label_order=self.config.label_order,
            )
        self.backend = backend

    # ------------------------------------------------------------------ API
    def verify(self, source: str, summary: str) -> VerificationReport:
        """K3'ün (kaynak, özet) çiftini denetler. K5 doğrudan bunu (veya pipeline.py'yi) çağırır."""
        summary = clean_summary(summary)
        cfg = self.config
        matches = split_and_match(
            source, summary, cfg.window_size, cfg.window_stride, cfg.top_k_windows
        )
        if not matches:
            return VerificationReport(summary=summary, results=[])

        source_numbers = extract_numbers(source) if cfg.check_numbers else set()

        # Tüm cümleler için çiftleri topla -> tek toplu NLI çağrısı
        pairs: List[Tuple[str, str]] = []
        spans: List[Tuple[int, int]] = []
        for m in matches:
            start = len(pairs)
            pairs.extend((w, m["hypothesis"]) for w in m["premises"])
            spans.append((start, len(pairs)))

        probs = self.backend.predict_proba(pairs)

        results = []
        for m, (a, b) in zip(matches, spans):
            results.append(
                self._decide(m["index"], m["hypothesis"], pairs[a:b], probs[a:b], source_numbers)
            )
        return VerificationReport(summary=summary, results=results)

    def verify_batch(self, items: Sequence[Tuple[str, str]]) -> List[VerificationReport]:
        """[(kaynak, özet), ...] -> raporlar. Toplu değerlendirme (K2 karşılaştırması vb.) için."""
        return [self.verify(src, summ) for src, summ in items]

    # ------------------------------------------------------------- internals
    def _decide(
        self,
        idx: int,
        hyp: str,
        pairs: Sequence[Tuple[str, str]],
        probs: Sequence[Probs],
        source_numbers: set,
    ) -> SentenceResult:
        cfg = self.config
        best_e = max(range(len(probs)), key=lambda j: probs[j][0])
        best_c = max(range(len(probs)), key=lambda j: probs[j][2])
        max_e, max_c = probs[best_e][0], probs[best_c][2]

        if max_e >= cfg.entailment_threshold:
            verdict, j = Verdict.ENTAILMENT, best_e
        elif max_c >= cfg.contradiction_threshold and max_c > max_e:
            verdict, j = Verdict.CONTRADICTION, best_c
        else:
            verdict, j = Verdict.NEUTRAL, best_e

        e, n, c = probs[j]
        confidence = {Verdict.ENTAILMENT: e, Verdict.NEUTRAL: n, Verdict.CONTRADICTION: c}[verdict]
        reason = None

        if cfg.check_numbers and verdict == Verdict.ENTAILMENT:
            missing = extract_numbers(hyp) - source_numbers
            if missing:
                verdict = Verdict.NEUTRAL
                confidence = 1.0  # kural tabanlı override
                reason = "numeric_mismatch: " + ", ".join(sorted(missing))

        return SentenceResult(
            index=idx,
            sentence=hyp,
            verdict=verdict,
            confidence=float(confidence),
            entailment=float(e),
            neutral=float(n),
            contradiction=float(c),
            evidence=pairs[j][0],
            reason=reason,
        )


# ---------------------------------------------------------------- K5 kolaylığı
_default: Optional[GuardrailVerifier] = None


def get_default_verifier(config: Optional[GuardrailConfig] = None) -> GuardrailVerifier:
    """Modeli süreç başına bir kez yükler (Streamlit her etkileşimde yeniden yüklemesin)."""
    global _default
    if _default is None:
        _default = GuardrailVerifier(config)
    return _default


def verify_summary(source: str, summary: str) -> VerificationReport:
    return get_default_verifier().verify(source, summary)
