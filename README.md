# Zsh Ollama Command Helper

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/vitali87/llm-shell)](https://github.com/vitali87/llm-shell/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/vitali87/llm-shell)](https://github.com/vitali87/llm-shell/issues)
[![GitHub forks](https://img.shields.io/github/forks/vitali87/llm-shell)](https://github.com/vitali87/llm-shell/network)
[![Contributors](https://img.shields.io/github/contributors/vitali87/llm-shell)](https://github.com/vitali87/llm-shell/graphs/contributors)
[![Last Commit](https://img.shields.io/github/last-commit/vitali87/llm-shell)](https://github.com/vitali87/llm-shell/commits/main)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

<a href="https://github.com/vitali87/llm-shell/stargazers"><img src="https://reporoster.com/stars/vitali87/llm-shell" alt="Stargazers"></a>

</div>

This project enhances your Zsh terminal by allowing you to input natural language queries for shell commands you can't remember. By pressing `Ctrl+B`, your query is sent to an Ollama model, which generates the appropriate command. The command is displayed, and you're prompted to execute it or not (`y/n`).

## 🎮 Demo



https://github.com/user-attachments/assets/64c6c2df-d8a4-4360-adcb-b381a0907f18



> 💡 Simply type your question and press `Ctrl+B` to get the command you need!

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

## ✨ Features

<table>
<tr>
<td>

### 🗣️ Natural Language Queries
Ask questions in plain English about shell commands
  
### 🤖 Model Integration
Uses your locally running Ollama instance with a finetuned model
  
### ⚡ Command Execution
Shows the generated command and prompts you to execute it

</td>
<td>

### 🔄 Model Selection
Easily switch between different Ollama models
  
### 🎨 Customizable
Adjust colors and default settings to your preference
  
### 🔒 Privacy-Focused
100% local execution - your queries never leave your machine

</td>
</tr>
</table>

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

## 📋 Roadmap & TODO

<div align="center">

![Progress](https://img.shields.io/badge/Progress-20%25-brightgreen)

</div>

### 🚀 Upcoming Features

#### User Experience
- [ ] 🎨 Add color themes support
  - [ ] Dark mode
  - [ ] Light mode
  - [ ] Terminal-native theme
- [ ] ⌨️ Customizable keyboard shortcuts
- [ ] 💾 Command history with search functionality
- [ ] 🔍 Auto-completion suggestions

#### AI/ML Enhancements
- [ ] 🧠 Context-aware command suggestions
- [ ] 📊 Learning from user corrections


#### Performance & Integration
- [ ] ⚡ Improve response time
- [ ] 🔌 Plugin system for extensions
- [ ] 📦 Package for different package managers
  - [ ] Homebrew
  - [ ] apt
  - [ ] pip

### 🔄 In Progress

#### User Feedback System
- [x] Basic feedback collection
- [ ] 👍 Command rating system (thumbs up/down)
- [ ] 📝 Feedback submission UI
- [ ] 📊 Analytics dashboard for feedback

#### Command History Enhancement
- [x] Basic history storage
- [ ] 🔍 Searchable command history
- [ ] 📈 Usage statistics
- [ ] 🎯 Success/failure tracking

### 🎯 Future Goals

#### Community Features
- [ ] 👥 Command sharing platform
- [ ] 🌟 Popular commands repository
- [ ] 🤝 Community contributions system

#### Documentation
- [ ] 📚 API documentation
- [ ] 🎥 Video tutorials
- [ ] 👩‍💻 Developer guide
- [ ] 🌍 Internationalization

### ✅ Completed
- [x] Basic command generation
- [x] Model selection interface
- [x] Installation script
- [x] Basic error handling


## Acknowledgments 

Ollama for providing the LLM serving platform.

OpenAI for the openai Python package.

Feel free to contribute to this project by submitting issues or pull requests.

## 📊 Project Stats

<div align="center">
  <img src="https://github-readme-stats.vercel.app/api/pin/?username=vitali87&repo=llm-shell&theme=dark" alt="Repo Card"/>
  
  <img src="https://github-profile-summary-cards.vercel.app/api/cards/profile-details?username=vitali87&theme=dark" alt="Profile Details"/>
</div>

## 👥 Contributors

<a href="https://github.com/vitali87/llm-shell/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=vitali87/llm-shell" />
</a>


## 🛠️ Built With

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Shell_Script-121011?style=for-the-badge&logo=gnu-bash&logoColor=white" alt="Shell"/>
  <img src="https://img.shields.io/badge/Zsh-121011?style=for-the-badge&logo=gnu-bash&logoColor=white" alt="Zsh"/>
</p>

## 🌟 Show your support

Give a ⭐️ if this project helped you!

<a href="https://buymeacoffee.com/vitali87">
  <img src="https://img.buymeacoffee.com/button-api/?text=Buy me a coffee&emoji=&slug=vitali87&button_colour=FFDD00&font_colour=000000&font_family=Cookie&outline_colour=000000&coffee_colour=ffffff" />
</a>

<div align="center">
  <img src="https://visitor-badge.laobi.icu/badge?page_id=vitali87.llm-shell" alt="visitors"/>
</div>
