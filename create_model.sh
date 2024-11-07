#!/bin/bash

# Exit on any error
set -e

# Check if directory path is provided as argument
if [ $# -ne 1 ]; then
    echo "Usage: $0 <directory_path>"
    echo "Example: $0 /path/to/shell-commands"
    exit 1
fi

# Define paths using provided directory
SCRIPT_DIR="$1"
# Get just the last part of the path
MODEL_NAME=$(basename "$SCRIPT_DIR")
MODEL_FILE="${SCRIPT_DIR}/unsloth.Q8_0.gguf"
MODELFILE_PATH="${SCRIPT_DIR}/Modelfile"

# Check if directory exists
if [ ! -d "$SCRIPT_DIR" ]; then
    echo "Error: Directory not found at ${SCRIPT_DIR}"
    exit 1
fi

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

# Create Ollama model using just the directory name
echo "Creating Ollama model '${MODEL_NAME}'..."
ollama create "$MODEL_NAME" -f "$MODELFILE_PATH"

echo "Model '${MODEL_NAME}' creation completed successfully!"