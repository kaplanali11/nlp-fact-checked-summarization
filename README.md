# Türkçe Metinlerde Halüsinasyonsuz Özetleme ve Çıkarımsal Denetim (Hallu-NLI)

Stanford CS224N (HALLU-NLI) esinli; uzun Türkçe haber ve duyuru metinlerini soyutlayarak (abstractive) özetleyen ve üretilen iddiaların doğruluğunu kaynak metin üzerinden Türkçe NLI modeliyle denetleyen iki aşamalı hibrit NLP sistemi.

## 👥 5 Kişilik Ekip Görev Dağılımı
* **K1 (Veri & Tokenizer Mühendisi):** `src/utils/` - MLSum, TR-News veri hazırlığı, tokenizasyon ve metin parçalama.
* **K2 (Baseline & Metrik Uzmanı):** `src/baseline/` - TextRank klasik özetleme ve ROUGE-1/2/L, BLEU değerlendirme hattı.
* **K3 (Seq2Seq Model Uzmanı):** `src/summarization/` - Türkçe T5 mimarisi, sıfır vuruş testleri ve fine-tuning.
* **K4 (NLI Güvenlik Denetçisi):** `src/guardrail/` - BERTurk NLI çıkarım motoru, anlamsal eşleme ve güven skorlama.
* **K5 (Arayüz & Raporlama):** `app/` - Streamlit canlı doğrulama paneli (doğrulanmış yeşil, halüsinasyon kırmızı).

## 🚀 Kurulum ve Çalıştırma
```bash
pip install -r requirements.txt
python src/summarization/inference.py
streamlit run app/app.py
```
