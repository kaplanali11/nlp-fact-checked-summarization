"""K3 -> K4 yapıştırıcısı: özetle, sonra denetle. K5 (arayüz) bunu tek çağrıyla kullanır.

`summarizer`, K3'ün `TurkishT5Summarizer`'ı gibi `summarize(text, **kw) -> str`
metodu olan herhangi bir nesne olabilir (duck typing; K3 koduna bağımlılık yok).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .schemas import VerificationReport
from .verifier import GuardrailVerifier, get_default_verifier


@dataclass
class FactCheckedSummary:
    source: str
    summary: str
    report: VerificationReport

    def to_dict(self) -> Dict[str, Any]:
        return {"source": self.source, "report": self.report.to_dict()}


def summarize_and_verify(
    summarizer: Any,
    text: str,
    verifier: Optional[GuardrailVerifier] = None,
    **generation_kwargs: Any,
) -> FactCheckedSummary:
    """Örn: summarize_and_verify(summarizer, text, max_length=128, min_length=25)

    NOT: K3 girdiyi 512 token'da kesiyor; özet yalnızca metnin başını görür.
    Denetim ise TAM kaynak metne karşı yapılır (özetin kapsamadığı kısım hatalı
    şekilde 'uydurma' sayılmaz, çünkü kontrol yönü özet -> kaynak).
    """
    verifier = verifier or get_default_verifier()
    summary = summarizer.summarize(text, **generation_kwargs)
    report = verifier.verify(text, summary)
    return FactCheckedSummary(source=text, summary=report.summary, report=report)
