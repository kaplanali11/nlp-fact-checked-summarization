"""K4 birim testleri. Gerçek model indirmez; sahte NLI backend ve sahte K3 kullanır."""
import json

from src.guardrail import (
    FactCheckNLI,
    GuardrailConfig,
    GuardrailVerifier,
    Verdict,
    split_and_match,
    split_sentences,
    summarize_and_verify,
)
from src.guardrail.berturk_nli import NLIBackend
from src.guardrail.sentence_matcher import build_windows
from src.guardrail.text_utils import clean_summary, content_stems, tr_lower


class KeywordBackend(NLIBackend):
    """Sözcük örtüşmesine bakan sahte NLI; karar mantığını test etmek için yeterli."""

    def __init__(self):
        self.calls = 0

    def predict_proba(self, pairs):
        self.calls += 1
        out = []
        for premise, hyp in pairs:
            hs, ps = content_stems(hyp), content_stems(premise)
            ratio = len(hs & ps) / len(hs) if hs else 0.0
            if "değil" in tr_lower(hyp) and "değil" not in tr_lower(premise) and ratio > 0.6:
                out.append((0.05, 0.1, 0.85))
            elif ratio >= 0.8:
                out.append((0.9, 0.08, 0.02))
            else:
                out.append((0.1, 0.8, 0.1))
        return out


def make(**kw):
    return GuardrailVerifier(GuardrailConfig(**kw), backend=KeywordBackend())


# K1'in data/samples.json içindeki gerçek örnek
SAMPLE = (
    "Dokuz Eylül Üniversitesi Rektörlüğü, Tınaztepe Yerleşkesi bünyesinde ileri teknoloji "
    "ve yapay zeka araştırma merkezinin temellerinin atıldığını duyurdu. Merkezin inşaatının "
    "önümüzdeki ay başlayacağı ve 2027 yılı sonunda tamamlanacağı bildirildi. Açılış "
    "protokolüne yalnızca mühendislik fakültesi öğretim üyeleri davet edildi."
)


# ---------------------------------------------------------------- splitter
def test_splitter_abbreviation_and_ordinal():
    s = split_sentences("Dr. Ahmet Yılmaz 5. kata çıktı. Sonra indi.")
    assert s == ["Dr. Ahmet Yılmaz 5. kata çıktı.", "Sonra indi."]


def test_splitter_decimal_and_date():
    assert len(split_sentences("Fiyat 3.5 lira oldu. Herkes şaşırdı!")) == 2
    assert len(split_sentences("Toplantı 11.11.2026 tarihinde yapıldı. Başkan konuştu.")) == 2


def test_splitter_initial_and_empty():
    assert len(split_sentences("M. Kemal geldi. Herkes alkışladı.")) == 2
    assert split_sentences("   ") == []


def test_splitter_no_final_period():
    # mT5 özetleri sık sık noktasız biter
    assert split_sentences("Merkez 2027'de açılacak") == ["Merkez 2027'de açılacak"]


def test_windows_cover_tail():
    sents = [f"Cümle {i}." for i in range(7)]
    assert build_windows(sents, size=3, stride=2)[-1].endswith("Cümle 6.")


def test_split_and_match_contract():
    m = split_and_match(SAMPLE, "Merkez 2027 sonunda bitecek. Davet kısıtlı.", window_size=2, top_k=1)
    assert [x["index"] for x in m] == [0, 1]
    assert all(len(x["premises"]) == 1 for x in m)
    assert split_and_match("", "Bir cümle.") == []
    assert split_and_match(SAMPLE, "") == []


# ------------------------------------------------------- K3 çıktısı uyumu
def test_clean_summary_strips_sentinels():
    assert clean_summary("<extra_id_0> Merkez  açıldı. </s><pad>") == "Merkez açıldı."


def test_ascii_folded_text_still_retrieves_and_entails():
    # K3'ün inference.py öz-testi ASCII'ye çevrilmiş Türkçe kullanıyor
    src = "Başkent Ankara'dır. Nüfusu fazladır. Hava soğuktur."
    r = make().verify(src, "Baskent Ankara'dir.")
    assert r.results[0].verdict == Verdict.ENTAILMENT


# ---------------------------------------------------------------- verifier
SRC = "Ankara Türkiye'nin başkentidir. Nüfusu beş milyonu aşıyor."


def test_entailed_is_green():
    r = make().verify(SRC, "Ankara Türkiye'nin başkentidir.")
    assert r.results[0].verdict == Verdict.ENTAILMENT
    assert r.results[0].ui_color == "green"
    assert r.faithfulness_score == 1.0


def test_unsupported_is_red_neutral():
    r = make().verify(SRC, "Ankara dünyanın en büyük şehridir.")
    assert r.results[0].verdict == Verdict.NEUTRAL
    assert r.results[0].is_hallucination


def test_contradiction():
    src = "Ali okula gitti ve dersi dinledi."
    r = make().verify(src, "Ali okula gitti ve dersi dinlemiş değil.")
    assert r.results[0].verdict == Verdict.CONTRADICTION


def test_numeric_guard_on_real_sample():
    ok = make().verify(SAMPLE, "Merkezin inşaatı 2027 yılı sonunda tamamlanacak.").results[0]
    assert ok.verdict == Verdict.ENTAILMENT
    bad = make().verify(SAMPLE, "Merkezin inşaatı 2028 yılı sonunda tamamlanacak.").results[0]
    assert bad.verdict == Verdict.NEUTRAL and "2028" in bad.reason
    off = make(check_numbers=False).verify(SAMPLE, "Merkezin inşaatı 2028 yılı sonunda tamamlanacak.")
    assert off.results[0].verdict == Verdict.ENTAILMENT


def test_faithfulness_score_and_single_batch_call():
    v = make()
    r = v.verify(SRC, "Ankara Türkiye'nin başkentidir. Ankara dünyanın en büyük şehridir.")
    assert r.faithfulness_score == 0.5
    assert len(r.hallucinated) == 1
    assert v.backend.calls == 1  # tüm cümleler tek toplu çağrıda


def test_retrieval_finds_relevant_window_in_long_source():
    filler = [f"Alakasız bir konu hakkında bilgi {i} verildi." for i in range(10)]
    key = "Belediye yeni metro hattını kasımda açtı."
    src = " ".join(filler[:8] + [key] + filler[8:])
    r = make(window_size=2, top_k_windows=2).verify(src, "Belediye yeni metro hattını kasımda açtı.")
    assert r.results[0].verdict == Verdict.ENTAILMENT
    assert "metro" in r.results[0].evidence


def test_empty_inputs():
    v = make()
    assert v.verify(SRC, "").is_empty
    assert v.verify("", "Bir cümle.").is_empty


def test_json_roundtrip():
    d = json.loads(make().verify(SRC, "Ankara Türkiye'nin başkentidir.").to_json())
    assert d["results"][0]["ui_color"] == "green"


# ------------------------------------------------- K3 arayüzü + FactCheckNLI
class FakeK3:
    """K3'ün TurkishT5Summarizer.summarize(text, max_length, min_length) imzasını taklit eder."""

    def summarize(self, text, max_length=128, min_length=25):
        return "<extra_id_0> Merkezin inşaatı 2027 yılı sonunda tamamlanacak. Rektör istifa etti."


def test_summarize_and_verify_end_to_end():
    out = summarize_and_verify(FakeK3(), SAMPLE, verifier=make(), max_length=64)
    assert out.summary.startswith("Merkezin")          # sentinel temizlendi
    assert [r.ui_color for r in out.report.results] == ["green", "red"]


def test_factcheck_nli_verify_claim_contract():
    class Stub(FactCheckNLI):
        def __init__(self):  # model yükleme
            pass

        def predict_proba(self, pairs):
            return [(0.1, 0.2, 0.7)]

    res = Stub().verify_claim("öncül", "hipotez")
    assert res["label"] == "contradiction" and abs(res["confidence"] - 0.7) < 1e-9
    assert set(res["probs"]) == {"entailment", "neutral", "contradiction"}
