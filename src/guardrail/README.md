# K4 — NLI Güvenlik Denetçisi (`src/guardrail/`)

Özet cümlelerini kaynak metinle (öncül, hipotez) eşleştirip **entailment / neutral / contradiction**
kararı üretir. Neutral ve contradiction = şüpheli (halüsinasyon) → arayüzde kırmızı.

## Dosyalar
| Dosya | İş |
|---|---|
| `berturk_nli.py` | `FactCheckNLI` (HF NLI modeli), `verify_claim()` + toplu `predict_proba()` |
| `sentence_matcher.py` | Türkçe cümle bölme, kaynak pencereleme, `split_and_match()` |
| `verifier.py` | Karar motoru (`GuardrailVerifier`), eşikler, sayı koruması |
| `pipeline.py` | `summarize_and_verify(k3_summarizer, text)` — K3 → K4 tek çağrı |
| `schemas.py` | K5'in tükettiği `VerificationReport` / `SentenceResult` |

## K3'ten beklenen girdi
`TurkishT5Summarizer.summarize(text) -> str`. K4 yalnızca `(kaynak: str, özet: str)` ister;
`<extra_id_N>`, `<pad>`, `</s>` artıkları otomatik temizlenir, noktasız biten özetler tek cümle sayılır.

## K5 için kullanım
```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # streamlit run app/app.py kökü görmez!

from src.guardrail import summarize_and_verify
from src.summarization.inference import TurkishT5Summarizer
import streamlit as st

@st.cache_resource
def load():
    from src.guardrail import GuardrailVerifier
    return TurkishT5Summarizer(), GuardrailVerifier()

summarizer, verifier = load()
out = summarize_and_verify(summarizer, text, verifier=verifier)
for r in out.report.results:
    r.ui_color      # "green" | "red"
    r.sentence, r.verdict, r.confidence, r.evidence, r.reason
out.report.faithfulness_score
```

## Çalıştırma
```bash
python -m pytest tests -q                                  # model indirmez
python -m src.guardrail.cli --source kaynak.txt --summary ozet.txt   # -m ile çalıştır (göreli import)
```

## Bilinen sınırlar
- Cümle-bazlı NLI birden çok kaynak cümlesini birleştiren çıkarımları kaçırır (yanlış kırmızı).
- Eşikler (0.60 / 0.50) kalibre edilmedi; validation setinde ayarla.
- NLI modeli çeviri veriyle (SNLI/MNLI-tr) eğitilmiş: haber/duyuru dilinde domain kayması var.
- Yalnızca sayı kuralı var; özel isim değişimi (Ali→Veli) kuralı yok.
