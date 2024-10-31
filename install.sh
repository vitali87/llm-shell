#!/usr/bin/env bash

# Create necessary directories
mkdir -p ~/.config/zsh/ollama_env/venv

# Check for Python 3.11
if command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN=python3.11
else
    echo "Python 3.11 is required. Please install it first."
    exit 1
fi

# Create virtual environment
$PYTHON_BIN -m venv ~/.config/zsh/ollama_env/venv

# Activate virtual environment and install dependencies
source ~/.config/zsh/ollama_env/venv/bin/activate
pip install openai

# Deactivate virtual environment
deactivate

# Create the Python script
cat << 'EOF' > ~/.config/zsh/ollama_env/ollama_helper.py
#!/usr/bin/env python3

import os
import sys
import json
import openai

def main():
    user_query = sys.argv[1]
    model = os.getenv("ZSH_OLLAMA_MODEL", "shell-commands:latest")
    openai.api_base = "http://localhost:11434/v1"

    response = openai.Completion.create(
        model=model,
        prompt=user_query,
        max_tokens=150,
        temperature=0,
    )

    if response and response.choices:
        command = response.choices[0].text.strip()
        if command:
            # Print the user's query and the generated command as JSON
            print(json.dumps({
                "user_query": user_query,
                "command": command
            }))
        else:
            print("Error: No command generated.", file=sys.stderr)
            sys.exit(1)
    else:
        print("Error: Failed to get a response from the model.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
EOF

chmod +x ~/.config/zsh/ollama_env/ollama_helper.py

# Add the Zsh configuration to ~/.zshrc if not already present
if ! grep -q "ollama_command_helper" ~/.zshrc; then
    echo "Adding ollama_command_helper to ~/.zshrc"
    cat << 'EOL' >> ~/.zshrc

# Ollama Command Helper Configuration
export ZSH_OLLAMA_MODEL="shell-commands:latest"

function set_ollama_model {
    if [[ $# -eq 0 ]]; then
        echo "Current model: $ZSH_OLLAMA_MODEL"
        echo "Available models:"
        ollama list | awk 'NR>1 {print $1}'
    else
        export ZSH_OLLAMA_MODEL="$1"
        echo "Ollama model set to: $ZSH_OLLAMA_MODEL"
    fi
}

compdef '_values "models" $(ollama list | awk '\''NR>1 {print $1}'\'')' set_ollama_model

function ollama_command_helper {
    local query="$BUFFER"
    BUFFER=""
    echo -e "\n🤔 \e[34mAsking Ollama (using model: $ZSH_OLLAMA_MODEL)...\e[0m"
    local result
    result=$(~/.config/zsh/ollama_env/venv/bin/python3 ~/.config/zsh/ollama_env/ollama_helper.py "$query" 2>/dev/null)
    if [[ $? -ne 0 ]]; then
        echo -e "\e[31m❌ No command generated.\e[0m"
        return 1
    fi
    local user_query command
    user_query=$(echo "$result" | jq -r '.user_query')
    command=$(echo "$result" | jq -r '.command')
    echo -e "\e[33mYour query:\e[0m $user_query"
    echo -e "\e[32mGenerated command:\e[0m $command"
    echo -n "Execute? [y/N] "
    read -r yn
    if [[ "$yn" == [Yy] ]]; then
        eval "$command"
    else
        echo "Aborted."
    fi
    zle reset-prompt
}
zle -N ollama_command_helper 
bindkey '^B' ollama_command_helper 
EOL 
fi 

echo "Setup complete. Please run 'source ~/.zshrc' to activate the changes."