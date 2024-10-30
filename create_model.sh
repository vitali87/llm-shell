#!/bin/bash

# Create Modelfile with proper template
cat > ./shell-commands/Modelfile << 'EOL'
FROM /home/vitali/Documents/code-finetune/shell-commands/unsloth.Q8_0.gguf

TEMPLATE """
### Instruction:
{{ .Prompt }}

### Response:
{{ .Response }}"""
EOL

# Create Ollama model
ollama create shell-commands -f ./shell-commands/Modelfile
