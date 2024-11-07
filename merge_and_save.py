import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import argparse

def merge_and_save_model(
    base_model_name="Qwen/Qwen2-1.5B",
    lora_weights="final_model_lora",  # Default to the new path
    output_dir="merged_model"
):
    print(f"Using LoRA weights from: {lora_weights}")
    print("Loading base model...")
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True
    )

    print("Loading LoRA weights from safetensors...")
    model = PeftModel.from_pretrained(
        model, 
        lora_weights,
        is_trainable=False,
        adapter_name="default"
    )
    
    print("Merging weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {output_dir} in safetensors format...")
    model.save_pretrained(
        output_dir, 
        safe_serialization=True,
    )
    tokenizer.save_pretrained(output_dir)
    
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_weights", type=str, default="final_model_lora",
                      help="Path to LoRA weights (default: final_model_lora)")
    args = parser.parse_args()
    
    merge_and_save_model(lora_weights=args.lora_weights)