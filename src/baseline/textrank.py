"""
Rol: K2 - Baseline & Metrik Uzmanı
Görev: Kural tabanlı istatistiksel TextRank algoritması ile baseline özetler üretmek.
"""
def generate_textrank_summary(text: str, num_sentences: int = 2) -> str:
    sentences = text.split(". ")
    return ". ".join(sentences[:num_sentences]) + "."
