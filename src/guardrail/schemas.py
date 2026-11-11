"""K3 -> K4 -> K5 arasındaki veri sözleşmesi.

K5 (arayüz) sadece bu sınıflara bağımlı olmalı; modelin iç detaylarına değil.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Verdict(str, Enum):
    ENTAILMENT = "entailment"          # kaynak cümleyi doğruluyor
    NEUTRAL = "neutral"                # kaynakta dayanağı yok (uydurma şüphesi)
    CONTRADICTION = "contradiction"    # kaynakla çelişiyor


@dataclass
class SentenceResult:
    index: int
    sentence: str
    verdict: Verdict
    # Nihai karar sınıfı için güven. Kural tabanlı override varsa (sayı kontrolü) 1.0.
    confidence: float
    # Kararı belirleyen pencerenin ham NLI olasılıkları
    entailment: float
    neutral: float
    contradiction: float
    # Kararı destekleyen kaynak pencere (arayüzde "kanıt" olarak gösterilebilir)
    evidence: str
    # Model dışı bir kural devreye girdiyse açıklaması (ör. "numeric_mismatch: 900")
    reason: Optional[str] = None

    @property
    def is_hallucination(self) -> bool:
        return self.verdict != Verdict.ENTAILMENT

    @property
    def ui_color(self) -> str:
        return "red" if self.is_hallucination else "green"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "sentence": self.sentence,
            "verdict": self.verdict.value,
            "is_hallucination": self.is_hallucination,
            "ui_color": self.ui_color,
            "confidence": round(self.confidence, 4),
            "entailment": round(self.entailment, 4),
            "neutral": round(self.neutral, 4),
            "contradiction": round(self.contradiction, 4),
            "evidence": self.evidence,
            "reason": self.reason,
        }


@dataclass
class VerificationReport:
    summary: str
    results: List[SentenceResult] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.results

    @property
    def faithfulness_score(self) -> float:
        """Doğrulanan cümle oranı. Boş özet için 0.0 döner; K5 is_empty'yi kontrol etmeli."""
        if not self.results:
            return 0.0
        ok = sum(1 for r in self.results if not r.is_hallucination)
        return ok / len(self.results)

    @property
    def hallucinated(self) -> List[SentenceResult]:
        return [r for r in self.results if r.is_hallucination]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "faithfulness_score": round(self.faithfulness_score, 4),
            "n_sentences": len(self.results),
            "n_hallucinated": len(self.hallucinated),
            "results": [r.to_dict() for r in self.results],
        }

    def to_json(self, **kwargs: Any) -> str:
        kwargs.setdefault("ensure_ascii", False)
        kwargs.setdefault("indent", 2)
        return json.dumps(self.to_dict(), **kwargs)
