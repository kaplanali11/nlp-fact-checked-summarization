"""
Rol: K3 - Seq2Seq Model Uzmani
Gorev: T5 modeli icin LoRA destekli egitim ve fine-tuning iskeleti (Hafta 8)
"""
from typing import Optional
from dataclasses import dataclass, field
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="dbmdz/turkish-t5-summarization",
        metadata={"help": "Temel alinacak T5 model adi"}
    )
    use_lora: bool = field(
        default=True,
        metadata={"help": "LoRA adaptoru kullanilip kullanilmayacagi"}
    )
    lora_r: int = field(default=8, metadata={"help": "LoRA rank degeri"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA olcekleme katsayisi"})

def setup_lora_model(base_model, lora_r: int = 8, lora_alpha: int = 32):
    """
    Hafif parametreli egitim icin LoRA adaptoru ekler.
    """
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.1,
            target_modules=["q", "v"]
        )
        return get_peft_model(base_model, peft_config)
    except ImportError:
        print("[K3] peft paketi yuklu degil, standart model donuluyor.")
        return base_model

def run_training_pipeline(
    train_dataset,
    eval_dataset,
    model_args: ModelArguments,
    training_args: Seq2SeqTrainingArguments
):
    """
    Hugging Face Seq2SeqTrainer ile model egitim dongusu.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path)

    if model_args.use_lora:
        model = setup_lora_model(model, lora_r=model_args.lora_r, lora_alpha=model_args.lora_alpha)

    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
    )
    return trainer

if __name__ == "__main__":
    print("[K3] Egitim modulu iskeleti hazir. Hafta 8 veri seti entegrasyonu bekleniyor.")
