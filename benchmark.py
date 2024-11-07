import argparse
import json
import sys
from typing import List, Dict, Any
import requests
from concurrent.futures import ThreadPoolExecutor
import time
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OllamaBatchProcessor:
    def __init__(
        self,
        model: str = "mistral",
        base_url: str = "http://localhost:11434",
        max_workers: int = 10,
        temperature: float = 0.0,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize the Ollama batch processor.
        
        Args:
            model: Name of the Ollama model to use
            base_url: Base URL for Ollama API
            max_workers: Maximum number of concurrent requests
            temperature: Temperature for response generation
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
        """
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.max_workers = max_workers
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def process_single_prompt(self, prompt: str, attempt: int = 0) -> Dict[str, Any]:
        """
        Process a single command instruction using the Ollama API.
        
        Args:
            prompt: The command instruction to process
            attempt: Current retry attempt number
            
        Returns:
            Dictionary containing the response data
        """
        headers = {
            "Content-Type": "application/json"
        }
        
        # Enhance the prompt for better command processing
        enhanced_prompt = f"Translate this command to its actual shell command(s). Return ONLY the command, no explanations: {prompt}"
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": enhanced_prompt}],
            "stream": False,
            "temperature": 0.1  # Lower temperature for more deterministic responses
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            return {
                "instruction": prompt,
                "command": result["message"]["content"].strip()
            }

        except (requests.RequestException, json.JSONDecodeError) as e:
            if attempt < self.max_retries:
                logger.warning(f"Retry {attempt + 1}/{self.max_retries} for prompt: {prompt[:50]}...")
                time.sleep(self.retry_delay)
                return self.process_single_prompt(prompt, attempt + 1)
            
            return {
                "instruction": prompt,
                "command": "ERROR: Failed to generate command"
            }

    def process_batch(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """
        Process a batch of prompts concurrently.
        
        Args:
            prompts: List of prompts to process
            
        Returns:
            List of dictionaries containing the results
        """
        logger.info(f"Processing {len(prompts)} prompts with {self.max_workers} workers")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(self.process_single_prompt, prompts))
            
        return results

def read_input_file(file_path: str) -> List[str]:
    """
    Read prompts from an input file. Each line is treated as a separate prompt.
    
    Args:
        file_path: Path to the input file
        
    Returns:
        List of prompts, one per line
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    if path.suffix.lower() == '.json':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and 'prompts' in data:
                return data['prompts']
            else:
                raise ValueError("JSON file must contain a list of prompts or a dict with 'prompts' key")
    else:
        # Process each line as a separate instruction
        prompts = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # Only add non-empty lines
                    prompts.append(line)
        
        logger.info(f"Loaded {len(prompts)} commands from text file")
        return prompts

def save_results(results: List[Dict[str, Any]], output_file: str):
    """
    Save the results to a file in JSON format.
    
    Args:
        results: List of result dictionaries
        output_file: Path to the output file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Process batch inputs using Ollama")
    parser.add_argument("input_file", help="Path to input file (JSON or text)")
    parser.add_argument("--output", "-o", default="output.json", help="Output file path")
    parser.add_argument("--model", "-m", default="mistral", help="Ollama model name")
    parser.add_argument("--workers", "-w", type=int, default=10, help="Number of worker threads")
    parser.add_argument("--temperature", "-t", type=float, default=0.0, help="Temperature for generation")
    parser.add_argument("--base-url", "-u", default="http://localhost:11434", help="Ollama API base URL")
    
    args = parser.parse_args()

    try:
        # Read input prompts
        prompts = read_input_file(args.input_file)
        logger.info(f"Loaded {len(prompts)} prompts from {args.input_file}")

        # Initialize processor
        processor = OllamaBatchProcessor(
            model=args.model,
            base_url=args.base_url,
            max_workers=args.workers,
            temperature=args.temperature
        )

        # Process prompts
        results = processor.process_batch(prompts)

        # Save results
        save_results(results, args.output)

        # Print simple summary
        logger.info(f"Processing complete. Generated commands for {len(results)} instructions.")

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
