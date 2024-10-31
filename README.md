# Zsh Ollama Command Helper

This project enhances your Zsh terminal by allowing you to input natural language queries for shell commands you can't remember. By pressing `Ctrl+B`, your query is sent to an Ollama model, which generates the appropriate command. The command is displayed, and you're prompted to execute it or not (`y/n`).

## Table of Contents
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)
- [TODO](#todo)
- [License](#license)

## Features
- **Natural Language Queries**: Ask questions in plain English about shell commands.
- **Model Integration**: Uses your locally running Ollama instance with a finetuned model.
- **Command Execution**: Shows the generated command and prompts you to execute it.
- **Model Selection**: Easily switch between different Ollama models.
- **Customizable**: Adjust colors and default settings to your preference.

## Prerequisites
- **Operating System**: Unix-like system (Linux, macOS).
- **Shell**: Zsh.
- **Python**: Version 3.11.
- **Ollama**: Installed and running locally.
- **jq**: Command-line JSON processor.

## Installation

### 1. Ensure Prerequisites Are Met

#### Install Zsh
If you don't have Zsh installed:

```bash
# On Ubuntu/Debian
sudo apt update
sudo apt install zsh

# On macOS (using Homebrew)
brew install zsh
```

#### Install Python 3.11
```bash
# On Ubuntu/Debian
sudo apt update
sudo apt install python3.11 python3.11-venv

# On macOS (using Homebrew)
brew install python@3.11
```

#### Install Ollama
Follow the installation instructions from Ollama's official documentation.

#### Install jq
```bash
# On Ubuntu/Debian
sudo apt update
sudo apt install jq

# On macOS (using Homebrew)
brew install jq
```

### 2. Run the `install.sh` Script

```bash 
./install.sh 
```

### 3. Reload Your Zsh Configuration 

```bash 
source ~/.zshrc 
```

## Configuration

### Default Ollama Model 

The default model is set to `vitali87/shell-commands`. You can change it by editing the `export ZSH_OLLAMA_MODEL` line in your `~/.zshrc`.

```bash 
export ZSH_OLLAMA_MODEL="your-preferred-model"
```

## Usage

### Ask a Question 

Type your natural language query directly into the terminal prompt.

```bash 
how to list all files modified in the last 24 hours 
```

### Trigger the Helper 

Press `Ctrl+B` to activate the helper.

#### Example Output:

```bash 
🤔 Asking Ollama (using model: vitali87/shell-commands)... 
Your query: how to list all files modified in the last 24 hours 
Generated command: find . -type f -mtime -1 
Execute? [y/N] 
```

### Execute the Command 

Press `y` and hit Enter to execute the command. Press any other key to abort.

## Customization

### Changing the Model 

#### List Available Models 

```bash 
set_ollama_model 
```

#### Set a Different Model 

```bash 
set_ollama_model your_model_name 
```

##### Example:

```bash 
set_ollama_model llama2:7b 
```

### Customizing Colors 

The colors can be adjusted by modifying the color codes in the Zsh configuration.

#### Color Codes:
- Black: `\e[30m`
- Red: `\e[31m`
- Green: `\e[32m`
- Yellow: `\e[33m`
- Blue: `\e[34m`
- Magenta: `\e[35m`
- Cyan: `\e[36m`
- White: `\e[37m`
- Reset: `\e[0m`

#### Steps:

Open the Zsh configuration file:

```bash 
nano ~/.zshrc 
```

Locate the `ollama_command_helper` function.

Modify the echo statements:

```bash 
echo -e "\e[33mYour query:\e[0m $user_query" 
echo -e "\e[32mGenerated command:\e[0m $command" 
```

Replace the color codes with your preferred ones.

Save and exit the editor.

Reload your Zsh configuration:

```bash 
source ~/.zshrc 
```

## Troubleshooting

### Error: No Command Generated 

Ensure your Ollama server is running and the specified model is available.

### Dependencies Not Found 

Make sure Python 3.11 and jq are installed on your system.

### Virtual Environment Issues 

If you encounter issues with the virtual environment:

Remove the existing virtual environment:

```bash 
rm -rf ~/.config/zsh/ollama_env 
```

Re-run the installation script:

```bash 
./install.sh 
```

Reload your Zsh configuration:

```bash 
source ~/.zshrc 
```

### Powerlevel10k Warning 

If you see a warning related to Powerlevel10k's instant prompt:

Place the Ollama Command Helper configuration after Powerlevel10k initialization in your `~/.zshrc`.

Alternatively, disable the instant prompt feature in Powerlevel10k.

## TODO

### Planned Features

#### User Feedback Integration
- [ ] Collect user feedback on generated commands
- [ ] Store feedback data for model improvement
- [ ] Implement feedback submission mechanism
- [ ] Create feedback analysis pipeline
- [ ] Periodically retrain model with user feedback
- [ ] Add command rating system (thumbs up/down)

#### Command History Enhancement
- [ ] Store history of queries and generated commands
- [ ] Implement context-aware command generation
- [ ] Add command suggestions based on previous usage
- [ ] Create searchable command history
- [ ] Save user modifications to generated commands
- [ ] Implement command success/failure tracking

### Implementation Details

#### Feedback System
- Store user feedback in `~/.config/zsh/ollama_feedback.jsonl`
- Track:
  - Original query
  - Generated command
  - User modifications (if any)
  - Execution success/failure
  - User rating
  - Timestamp

#### Command History
- Store command history in `~/.config/zsh/ollama_history.jsonl`
- Include:
  - Query-command pairs
  - Execution context
  - Success rate
  - Usage frequency
  - Related commands

## License 

This project is licensed under the MIT License.

---

Note: Replace `vitali87/shell-commands` with the name of your finetuned Ollama model if it's different.

## Additional Information

### Verifying Ollama is Running 

To ensure that Ollama is running and necessary models are available:

```bash 
ollama list 
```

#### Example Output:

```bash  
NAME                        ID              SIZE      MODIFIED  
vitali87/shell-commands       abcdef123456     5.4 GB   2 days ago  
qwen2.5-coder:latest       123456abcdef     4.7 GB   3 weeks ago  
```

## Acknowledgments 

Ollama for providing the LLM serving platform.

OpenAI for the openai Python package.

Feel free to contribute to this project by submitting issues or pull requests.



