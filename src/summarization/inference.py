"""
Rol: K3 - Seq2Seq Model Uzmanı
Görev: Önceden eğitilmiş Türkçe T5 modeli ile soyutlayıcı özet çıkarma (Hafta 6-7)
"""
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class TurkishT5Summarizer:
    def __init__(self, model_name="ozcangundes/mt5-small-turkish-summarization", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[K3] Model yukleniyor: {model_name} ({self.device})")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def summarize(self, text: str, max_length: int = 128, min_length: int = 25) -> str:
        input_text = "özetle: " + text.strip()
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(self.device)
        with torch.no_grad():
            summary_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                min_length=min_length,
                num_beams=4,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    summarizer = TurkishT5Summarizer()
    test_text = "Dokuz Eylul Universitesi Rektörlügü, Tinaztepe Yerleskesi bunyesinde yeni yapay zeka arastirma merkezi acilacagini duyurdu. Insaat onumuzdeki ay baslayacak ve 2027 sonunda tamamlanacak. Acilis torenine yalnizca muhendislik fakultesi ogretim uyeleri davet edildi."
    print("\n--- TEST OZETI ---")
    print(summarizer.summarize(test_text))
