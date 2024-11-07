import os
import torch
import torch.distributed as dist
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from trl import SFTTrainer

def setup_distributed():
    """Set up distributed training."""
    if int(os.environ.get("LOCAL_RANK", -1)) != -1:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group(backend='nccl')

def load_model_and_tokenizer(model_name, local_rank):
    """Load and configure the model and tokenizer."""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with specific configuration
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": local_rank} if torch.cuda.is_available() else "cpu"
    )
    
    return model, tokenizer

def prepare_model_for_training(model, local_rank):
    """Prepare the model for training with LoRA."""
    # First prepare for k-bit training
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # THEN configure and apply LoRA 
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    
    # Enable input requires grad
    model.enable_input_require_grads()
    
    # Disable the cache to save memory
    model.config.use_cache = False
    
    model.train()  # Add this
    
    # Log trainable parameters
    if local_rank <= 0:
        model.print_trainable_parameters()
    
    return model
def preprocess_dataset(dataset, tokenizer, max_length=2048):
    """Preprocess the dataset with chat template."""
    def preprocess_function(example):
        # Create chat messages
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
        
        # Add EOS token
        prompt = prompt + tokenizer.eos_token
        
        # Tokenize
        tokenized = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors=None,
        )
        
        # Set labels
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    # Process the entire dataset
    processed_dataset = dataset.map(
        preprocess_function,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
    )

    return processed_dataset

def get_training_arguments(local_rank):
    """Configure training arguments."""
    return TrainingArguments(
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
        deepspeed="ds_config.json",
        local_rank=local_rank,
        ddp_find_unused_parameters=False,
        # Additional settings
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        torch_compile=False,
    )

def main():
    # Set up distributed training
    setup_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    
    # Configuration
    model_name = "Qwen/Qwen2-1.5B"
    max_seq_length = 2048
    dataset_path = 'data.jsonl'
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name, local_rank)
    
    # Prepare model for training (LoRA + optimization settings)
    model = prepare_model_for_training(model, local_rank)

    # Load and preprocess dataset
    dataset = load_dataset('json', data_files=dataset_path, split='train')
    processed_dataset = preprocess_dataset(dataset, tokenizer, max_seq_length)

    # Get training arguments
    training_args = get_training_arguments(local_rank)

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=processed_dataset,
        args=training_args,
        max_seq_length=max_seq_length,
        packing=False,
    )

    # Start training
    if local_rank <= 0:
        print("Starting training...")
    trainer.train()

    # Save the model (only on main process)
    if local_rank <= 0:
        print("Saving final model...")
        # Save the final checkpoint explicitly
        model.save_pretrained("final_model_lora", safe_serialization=True)
        tokenizer.save_pretrained("final_model_lora")
        print("Model saved to final_model_lora/")

    dist.destroy_process_group()  # Add this line
    
    # Optional: Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()