#!/bin/bash

# Create a temporary file
TEMP_FILE=$(mktemp)

# Read the JSONL file line by line, normalize each JSON object 
# (by sorting keys and removing whitespace) and keep unique lines
jq -c . data.jsonl | \
  while read -r line; do
    echo "$(echo "$line" | jq -cS '.')"
  done | sort | uniq > "$TEMP_FILE"

# Move temp file to original
mv "$TEMP_FILE" data.jsonl

echo "Duplicates removed. Check data.jsonl"
