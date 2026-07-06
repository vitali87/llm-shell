from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastModel

# Qwen3.5 is a vision-language model: it MUST be loaded with FastModel (not
# FastLanguageModel, which mis-patches the GatedDeltaNet layer), and the text
# tokenizer extracted via processor.tokenizer.
model, processor = FastModel.from_pretrained(
    "Qwen/Qwen3.5-2B",
    max_seq_length=2048,
    load_in_4bit=False,  # unsloth advises against 4-bit QLoRA for Qwen3.5; use bf16
    load_in_16bit=True,
    full_finetuning=False,
)
tokenizer = getattr(processor, "tokenizer", processor)

# Configure tokenizer
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Add LoRA adapters (text-only task: language layers, skip the vision tower)
model = FastModel.get_peft_model(
    model,
    r=8,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    use_gradient_checkpointing="unsloth",
)

# Load dataset
dataset = load_dataset("json", data_files="data.jsonl", split="train")


# Prepare dataset for training with chat format
def preprocess_function(example):
    # Format chat messages
    messages = [
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]},
    ]

    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    # Add EOS token explicitly
    prompt = prompt + tokenizer.eos_token

    # Tokenize with padding and truncation
    tokenized = tokenizer(
        prompt,
        truncation=True,
        max_length=2048,
        padding="max_length",
        return_tensors=None,
    )

    # Add labels for supervised fine-tuning, masking padding so the loss
    # ignores the pad tokens that fill each example up to max_length.
    tokenized["labels"] = [
        token if mask == 1 else -100
        for token, mask in zip(tokenized["input_ids"], tokenized["attention_mask"])
    ]

    # Also mask the prompt prefix so the loss trains only on the response.
    prompt_only = tokenizer.apply_chat_template(
        messages[:1], tokenize=False, add_generation_prompt=True
    )
    prompt_len = len(tokenizer(prompt_only, add_special_tokens=False).input_ids)
    for i in range(min(prompt_len, len(tokenized["labels"]))):
        tokenized["labels"][i] = -100

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
    num_train_epochs=10,
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
model.save_pretrained_gguf("shell-commands-qwen3.5-2b", tokenizer)
