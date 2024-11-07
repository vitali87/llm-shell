#!/bin/bash

# Exit on any error
set -e

# Check if a model name is provided
if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Usage: $0 <model_name> [lora_weights_path]"
    echo "Example: $0 shell-commands-qwen2-1.5b"
    echo "Example with custom weights: $0 shell-commands-qwen2-1.5b outputs/checkpoint-1100"
    exit 1
fi

MODEL_NAME="$1"
LORA_WEIGHTS="${2:-final_model_lora}"  # Use second argument if provided, else default to final_model_lora

echo "Starting post-training processing for model: $MODEL_NAME"
echo "Using LoRA weights from: $LORA_WEIGHTS"

# Create a directory for the final model if it doesn't exist
mkdir -p "$MODEL_NAME"

# Step 1: Merge LoRA weights with base model
echo "Step 1: Merging LoRA weights with base model..."
python merge_and_save_model.py --lora_weights "$LORA_WEIGHTS"

# Step 2: Convert to GGUF format
echo "Step 2: Converting to GGUF format..."
python llama.cpp/convert_hf_to_gguf.py merged_model --outfile "$MODEL_NAME/model.q8_0.gguf"

# Step 3: Create Modelfile
echo "Step 3: Creating Modelfile..."
cat > "$MODEL_NAME/Modelfile" << 'EOL'
FROM ./model.q8_0.gguf

PARAMETER temperature 0
PARAMETER top_p 0.7

TEMPLATE """
{{ if .System }}system: {{ .System }}{{ end }}
user: {{ .Prompt }}
assistant: """
EOL

# Step 4: Create Ollama model
echo "Step 4: Creating Ollama model..."
ollama create "$MODEL_NAME" -f "$MODEL_NAME/Modelfile"

# Cleanup intermediate files (optional)
echo "Cleaning up..."
rm -rf merged_model  # Remove the merged model directory

echo "Post-training processing complete!"
echo "You can now use: ollama run $MODEL_NAME"