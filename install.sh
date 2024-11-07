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
from openai import OpenAI
import traceback

def main():
    try:
        user_query = sys.argv[1]
        model = os.getenv("ZSH_OLLAMA_MODEL", "shell-commands:latest")
        
        # Debug: Print current settings
        print(f"Debug - Query: {user_query}", file=sys.stderr)
        print(f"Debug - Model: {model}", file=sys.stderr)
        
        # Create client with base URL for Ollama
        client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="no-key-needed"  # Ollama doesn't need an API key
        )
        
        try:
            print("Debug - Making API call...", file=sys.stderr)
            response = client.completions.create(
                model=model,
                prompt=user_query,
                max_tokens=150,
                temperature=0,
            )
            print(f"Debug - Raw Response: {response}", file=sys.stderr)
            
        except Exception as e:
            print(f"Debug - API Error: {str(e)}", file=sys.stderr)
            print(f"Debug - Traceback: {traceback.format_exc()}", file=sys.stderr)
            sys.exit(1)
        
        if hasattr(response, 'choices') and response.choices:
            command = response.choices[0].text
            if command:
                # Print the user's query and the generated command as JSON
                result = {
                    "user_query": user_query,
                    "command": command.strip()
                }
                print(f"Debug - Generated result: {result}", file=sys.stderr)
                print(json.dumps(result))
            else:
                print("Error: No command text in response", file=sys.stderr)
                sys.exit(1)
        else:
            print("Error: No choices in response", file=sys.stderr)
            print(f"Debug - Response structure: {dir(response)}", file=sys.stderr)
            sys.exit(1)
            
    except Exception as e:
        print(f"Debug - Unexpected error: {str(e)}", file=sys.stderr)
        print(f"Debug - Traceback: {traceback.format_exc()}", file=sys.stderr)
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
export ZSH_OLLAMA_MODEL="vitali87/shell-commands-qwen2-1.5b:latest"

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
    
    local yn
    while true; do
        echo -n "Execute? [y/N] "
        read -k 1 yn
        case $yn in
            [Yy]* ) 
                echo
                eval "$command"
                break
                ;;
            [Nn]* | $'\n' )
                echo
                echo "Aborted."
                break
                ;;
            * )
                ;;
        esac
    done
    
    zle reset-prompt
}

zle -N ollama_command_helper
bindkey '^B' ollama_command_helper   
EOL
fi

echo "Setup complete. Please run 'source ~/.zshrc' to activate the changes."