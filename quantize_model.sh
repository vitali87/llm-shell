#!/bin/bash

# Check if a directory argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <directory-path>"
  exit 1
fi

# Use the provided directory path
DIR="$1"

# Run the quantization command with the user-supplied directory
llama.cpp/llama-quantize "$DIR/qwen2.gguf" "$DIR/qwen2-1.5b-q8_0.gguf" Q8_0
