#!/bin/bash

# Remove large training artifacts that must not be committed.
# Mirrors the generated paths in .gitignore.

set -e

for path in outputs logs merged_model final_model_lora _unsloth_sentencepiece_temp shell-commands*; do
    rm -rf "$path"
done

echo "Cleaned training artifacts."
