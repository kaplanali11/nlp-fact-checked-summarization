"""
Rol: K4 - NLI Güvenlik Denetçisi
Görev: Özet cümlelerini kaynak metinle doğrulayıp Entailment / Neutral / Contradiction kararı vermek.

`NLIBackend` soyut arayüzü sayesinde karar motoru (verifier.py) modelden bağımsız
test edilebilir; testlerde gerçek model indirmeden sahte backend kullanılır.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple

from .config import GuardrailConfig

Probs = Tuple[float, float, float]  # (entailment, neutral, contradiction)
LABELS = ("entailment", "neutral", "contradiction")


class NLIBackend(ABC):
    @abstractmethod
    def predict_proba(self, pairs: Sequence[Tuple[str, str]]) -> List[Probs]:
        """Her (öncül, hipotez) çifti için (entailment, neutral, contradiction) olasılıkları."""


class FactCheckNLI(NLIBackend):
    """Hugging Face sequence-classification tabanlı (BERTurk vb.) NLI motoru.

    Eski iskeletteki `verify_claim(premise, hypothesis) -> {"label", "confidence"}`
    sözleşmesi korunur (ek olarak "probs" anahtarı döner).
    """

    def __init__(
        self,
        model_name: str = GuardrailConfig.model_name,
        device: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 16,
        label_order: Optional[Sequence[str]] = None,
    ) -> None:
        import torch  # tembel import: testler torch istemesin
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.batch_size = batch_size
        print(f"[K4] NLI modeli yukleniyor: {model_name} ({self.device})")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device).eval()
        self._idx = self._resolve_labels(self.model.config.id2label, label_order)

    @staticmethod
    def _resolve_labels(
        id2label: Dict[int, str], label_order: Optional[Sequence[str]]
    ) -> Tuple[int, int, int]:
        if label_order is not None:
            names = {i: n for i, n in enumerate(label_order)}
        else:
            names = {int(i): n for i, n in id2label.items()}

        found: Dict[str, int] = {}
        for i, name in names.items():
            low = name.lower()
            if low.startswith("entail"):
                found["e"] = i
            elif low.startswith("neutral"):
                found["n"] = i
            elif low.startswith("contradict"):
                found["c"] = i
        if len(found) != 3:
            raise ValueError(
                f"NLI etiketleri çözülemedi: {names}. "
                "GuardrailConfig.label_order ile indeks sırasını elle ver."
            )
        return found["e"], found["n"], found["c"]

    def predict_proba(self, pairs: Sequence[Tuple[str, str]]) -> List[Probs]:
        torch = self._torch
        e, n, c = self._idx
        out: List[Probs] = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i + self.batch_size]
            premises = [p for p, _ in batch]
            hyps = [h for _, h in batch]
            try:
                enc = self.tokenizer(
                    premises, hyps, truncation="only_first",
                    max_length=self.max_length, padding=True, return_tensors="pt",
                )
            except ValueError:  # hipotezin kendisi max_length'i aşıyorsa
                enc = self.tokenizer(
                    premises, hyps, truncation="longest_first",
                    max_length=self.max_length, padding=True, return_tensors="pt",
                )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.inference_mode():
                logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().tolist()
            out.extend((row[e], row[n], row[c]) for row in probs)
        return out

    def verify_claim(self, premise: str, hypothesis: str) -> dict:
        """Tek (öncül, hipotez) çifti için argmax etiket. Eşik uygulamaz; eşikler verifier.py'de."""
        probs = self.predict_proba([(premise, hypothesis)])[0]
        label, conf = max(zip(LABELS, probs), key=lambda x: x[1])
        return {"label": label, "confidence": float(conf), "probs": dict(zip(LABELS, probs))}
