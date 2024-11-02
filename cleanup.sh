from unsloth import FastLanguageModel, apply_chat_template
import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer

# Load model and tokenizer with 4-bit quantization
model, tokenizer = FastLanguageModel.from_pretrained(
    "Qwen/Qwen2-1.5B",
    max_seq_length=1024,
    load_in_4bit=True,
)

# Configure tokenizer
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                   "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    use_gradient_checkpointing=True,
)

# Load dataset
dataset = load_dataset('json', data_files='data.jsonl', split='train')

# Prepare dataset for training with chat format
def preprocess_function(example):
    # Format chat messages
    messages = [
        {"role": "user", "content": example['instruction']},
        {"role": "assistant", "content": example['output']}
    ]
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    
    # Add EOS token explicitly
    prompt = prompt + tokenizer.eos_token
    
    # Tokenize with padding and truncation
    tokenized = tokenizer(
        prompt,
        truncation=True,
        max_length=2048,
        padding='max_length',
        return_tensors=None,
    )
    
    # Add labels for supervised fine-tuning
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

# Process the dataset
processed_dataset = dataset.map(
    preprocess_function,
    remove_columns=dataset.column_names,
    desc="Tokenizing dataset",
)

# Update training arguments
training_arguments = TrainingArguments(
    output_dir="outputs",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    bf16=True,
    logging_steps=1,
    optim="adamw_torch_fused",
    save_strategy="steps",
    save_steps=50,
    logging_dir="logs",
    group_by_length=True,
    warmup_ratio=0.05,
    gradient_checkpointing=True,
    report_to="none",
)

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=processed_dataset,
    args=training_arguments,
    max_seq_length=2048,
    packing=False,
)

print("Starting training...")
trainer.train()

# Save the model
print("Saving model...")
model.save_pretrained_gguf("shell-commands", tokenizer)
print("Training completed!")