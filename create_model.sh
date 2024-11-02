#!/bin/bash

# Exit on any error
set -e

# Define paths
SCRIPT_DIR="$(pwd)/shell-commands"
MODEL_FILE="${SCRIPT_DIR}/unsloth.Q8_0.gguf"
MODELFILE_PATH="${SCRIPT_DIR}/Modelfile"

# Check if the model file exists
if [ ! -f "$MODEL_FILE" ]; then
    echo "Error: Model file not found at ${MODEL_FILE}"
    exit 1
fi

# Create Modelfile with proper template
# Using weak quotes for heredoc to allow variable expansion
cat > "$MODELFILE_PATH" << EOL
FROM ${MODEL_FILE}

PARAMETER temperature 0
PARAMETER top_p 0.7
PARAMETER stop "<|im_end|>"

TEMPLATE """
{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>{{ end }}<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""
EOL

# Check if Modelfile was created successfully
if [ ! -f "$MODELFILE_PATH" ]; then
    echo "Error: Failed to create Modelfile"
    exit 1
fi

echo "Created Modelfile successfully at ${MODELFILE_PATH}"

# Create Ollama model
echo "Creating Ollama model..."
ollama create shell-commands -f "$MODELFILE_PATH"

echo "Model creation completed successfully!"