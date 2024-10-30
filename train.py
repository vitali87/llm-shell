from unsloth import FastLanguageModel, apply_chat_template
import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer

# Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    "Qwen/Qwen2-1.5B", # Qwen/Qwen1.5-0.5B
    max_seq_length=1024,
    load_in_4bit=True
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                   "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0
)

# Load dataset
dataset = load_dataset('json', data_files='data.jsonl', split='train')

# Prepare dataset for training
def format_instruction(example):
    return {
        "text": f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    }

dataset = dataset.map(format_instruction)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    args=TrainingArguments(
        output_dir="outputs",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1
    )
)

trainer.train()
model.save_pretrained_gguf("shell-commands", tokenizer)
