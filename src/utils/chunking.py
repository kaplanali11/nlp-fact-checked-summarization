"""
Rol: K1 - Veri & Tokenizer Mühendisi
Görev: Uzun metinleri bağlam penceresini aşmayacak şekilde cümle/paragraf bazlı bölmek.
"""
def chunk_text(text: str, max_words: int = 250) -> list:
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]
