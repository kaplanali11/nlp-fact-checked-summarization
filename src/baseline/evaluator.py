"""
Rol: K2 - Baseline & Metrik Uzmanı
Görev: Referans özetler ile model çıktıları arasında ROUGE ve BLEU metriklerini hesaplamak.
"""
def compute_metrics(predictions: list, references: list) -> dict:
    return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
